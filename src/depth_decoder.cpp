#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace breeze {

static std::vector<float> llama3_freq_factors(const DepthConfig & c) {
    const int half = c.head_dim / 2;
    std::vector<float> ff(half, 1.0f);
    const float pi = 3.14159265358979323846f;
    const float low_wl = c.rope_orig_ctx / c.rope_low_freq;
    const float high_wl = c.rope_orig_ctx / c.rope_high_freq;
    for (int i = 0; i < half; i++) {
        float freq = std::pow(c.rope_theta, -2.0f * i / c.head_dim);
        float wavelen = 2.0f * pi / freq;
        if (wavelen > low_wl) {
            ff[i] = c.rope_factor;
        } else if (wavelen < high_wl) {
            ff[i] = 1.0f;
        } else {
            float smooth = (c.rope_orig_ctx / wavelen - c.rope_low_freq) / (c.rope_high_freq - c.rope_low_freq);
            ff[i] = 1.0f / ((1.0f - smooth) / c.rope_factor + smooth);
        }
    }
    return ff;
}

static void check_shape(ggml_tensor * t, const std::string & name,
                        int64_t n0, int64_t n1, int64_t n2 = -1) {
    if (t->ne[0] == n0 && t->ne[1] == n1 && (n2 < 0 || t->ne[2] == n2)) return;
    throw std::runtime_error(
        name + " is [" + std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) + ", " +
        std::to_string(t->ne[2]) + "], expected [" + std::to_string(n0) + ", " + std::to_string(n1) +
        ", " + (n2 < 0 ? std::string("any") : std::to_string(n2)) + "]");
}

void DepthRunner::init(BreezeModel & m, int n_branches) {
    n_branch = n_branches;
    const DepthConfig & c = m.cfg.dd;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;

    // the depth graph slices into these by codebook, so a gguf built for a different codebook or
    // vocab count reads off the end and comes out as noise instead of failing
    check_shape(m.w("dd.in_proj.weight"), "dd.in_proj.weight", m.cfg.hidden_size, c.hidden);
    check_shape(m.w("audio_embd.weight"), "audio_embd.weight", m.cfg.hidden_size, (int64_t) nc * vs);
    check_shape(m.w("dd.codebooks_head.weight"), "dd.codebooks_head.weight", c.hidden, vs, nc - 1);

    kv.init(m.backend, c.n_layer, c.head_dim, c.n_kv_head, nc + 1, n_branches);
    freq_factors = llama3_freq_factors(c);
}

void DepthRunner::free() {
    kv.free();
}

static ggml_tensor * dd_layer(ggml_context * ctx, BreezeModel & m, Graph & g, KVCache & kv,
                              ggml_tensor * x, int il, ggml_tensor * pos, ggml_tensor * ff,
                              ggml_tensor * mask, int start, int n) {
    const DepthConfig & c = m.cfg.dd;
    const std::string p = "dd.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    q = ggml_rope_ext(ctx, q, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kfull = cache_append(ctx, g, kv.k[il], k, start);
    ggml_tensor * vfull = cache_append(ctx, g, kv.v[il], v, start);
    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

// runs one depth position for every branch at once and returns the per branch logits,
// laid out branch major so branch b starts at b * vocab
static std::vector<float> depth_step(BreezeModel & m, DepthRunner & r, int start,
                                     const std::vector<std::vector<float>> * hiddens,
                                     const std::vector<int> & audio_codes, int head_idx, int nb) {
    const DepthConfig & c = m.cfg.dd;
    const int n_pos = hiddens ? 2 : 1;
    const int n_tok = n_pos * nb;
    const int total = (start + n_pos) * nb;
    Graph g(4096);

    std::vector<int32_t> idx(audio_codes.begin(), audio_codes.end());
    ggml_tensor * aud = g.input_i32(idx, nb);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), aud); // [2048, nb]
    if (hiddens) {
        std::vector<float> flat;
        flat.reserve((size_t) nb * m.cfg.hidden_size);
        for (const auto & h : *hiddens) flat.insert(flat.end(), h.begin(), h.end());
        ggml_tensor * h0 = g.input_f32(flat, m.cfg.hidden_size, nb);
        embed = ggml_concat(g.ctx, h0, embed, 1); // [2048, 2*nb], position major
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, n_tok]

    std::vector<int32_t> pos_i(n_tok);
    for (int i = 0; i < n_tok; i++) pos_i[i] = start + i / nb;
    ggml_tensor * pos = g.input_i32(pos_i, n_tok);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());
    std::vector<float> mask_v = build_branch_causal_mask(n_tok, total, start, nb);
    ggml_tensor * mask = g.input_f32(mask_v, total, n_tok);

    for (int il = 0; il < c.n_layer; il++)
        x = dd_layer(g.ctx, m, g, r.kv, x, il, pos, ff, mask, start * nb, n_tok);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, nb, x->nb[1],
                                                       (size_t) (n_tok - nb) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1], (size_t) head_idx * head->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(g.ctx, hw, last); // [vocab, nb]
    g.compute(m.backend, logits);
    return tensor_to_f32(logits);
}

std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng,
                                  const SampleParams * sp_in, const int * force, int n_force) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    kv.reset();

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;
    if (sp_in) sp = *sp_in;

    std::vector<int> codes = { cb0 };
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        std::vector<int> cur_codes(n_branch, j == 1 ? cb0 : codes[head_idx] + head_idx * vs);
        std::vector<float> out = j == 1
            ? depth_step(m, *this, 0, &hiddens, cur_codes, head_idx, n_branch)
            : depth_step(m, *this, j, nullptr, cur_codes, head_idx, n_branch);

        const int vocab = (int) out.size() / n_branch;
        std::vector<float> logits(out.begin(), out.begin() + vocab);
        if (n_branch > 1) {
            for (int i = 0; i < vocab; i++)
                logits[i] = out[vocab + i] + cfg_scale * (out[i] - out[vocab + i]);
        }
        // forced steps still run the graph, later codebooks are conditioned on this one
        codes.push_back(j <= n_force ? force[j - 1] : sample_token(logits, sp, rng));
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

std::vector<std::vector<int>> DepthRunner::run_batched(BreezeModel & m, const std::vector<BatchItem> & items) {
    if (items.empty()) return {};
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    kv.reset();

    // 1. Map items to branches
    struct BranchInfo {
        size_t item_idx;
        bool is_uncond;
    };
    std::vector<BranchInfo> b_info;
    std::vector<int> cur_audio_codes;
    std::vector<std::vector<float>> all_hiddens;

    for (size_t i = 0; i < items.size(); i++) {
        const auto & it = items[i];
        // Conditional branch
        b_info.push_back({ i, false });
        cur_audio_codes.push_back(it.cb0);
        all_hiddens.push_back(it.hidden_c);

        // Unconditional branch (if CFG enabled)
        if (it.use_cfg) {
            b_info.push_back({ i, true });
            cur_audio_codes.push_back(it.cb0);
            all_hiddens.push_back(it.hidden_u);
        }
    }

    const int total_b = (int) b_info.size();
    if (total_b > n_branch) {
        kv.free();
        init(m, std::max(total_b, n_branch * 2));
    }

    std::vector<std::vector<int>> results(items.size());
    for (size_t i = 0; i < items.size(); i++) {
        results[i].reserve(nc - 1);
    }

    // 2. 15 Depth Steps (Batched GEMM)
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        std::vector<float> out = (j == 1)
            ? depth_step(m, *this, 0, &all_hiddens, cur_audio_codes, head_idx, total_b)
            : depth_step(m, *this, j, nullptr, cur_audio_codes, head_idx, total_b);

        const int vocab = (int) out.size() / total_b;

        // Unpack per item
        size_t b_idx = 0;
        for (size_t i = 0; i < items.size(); i++) {
            const auto & it = items[i];
            SampleParams sp;
            sp.temperature = m.cfg.depth_temperature;
            sp.top_k = m.cfg.depth_top_k;
            sp.top_p = m.cfg.depth_top_p;
            if (it.sp) sp = *it.sp;

            int sampled = -1;
            if (it.use_cfg) {
                const float * lc = out.data() + b_idx * vocab;
                const float * lu = out.data() + (b_idx + 1) * vocab;
                std::vector<float> comb_logits(vocab);
                for (int v = 0; v < vocab; v++) {
                    comb_logits[v] = lu[v] + it.cfg_scale * (lc[v] - lu[v]);
                }
                sampled = sample_token(comb_logits, sp, *it.rng);
                cur_audio_codes[b_idx] = sampled + head_idx * vs;
                cur_audio_codes[b_idx + 1] = sampled + head_idx * vs;
                b_idx += 2;
            } else {
                const float * lc = out.data() + b_idx * vocab;
                std::vector<float> logits(lc, lc + vocab);
                sampled = sample_token(logits, sp, *it.rng);
                cur_audio_codes[b_idx] = sampled + head_idx * vs;
                b_idx += 1;
            }
            results[i].push_back(sampled);
        }
    }

    return results;
}

std::vector<std::vector<int>> DepthRunner::run_batched_unified_gpu(BreezeModel & m, const std::vector<BatchItem> & items) {
    if (items.empty()) return {};

    // If any item uses CFG, gracefully fallback to run_batched to preserve complex multi-branch interpolation
    for (const auto & it : items) {
        if (it.use_cfg) {
            return run_batched(m, items);
        }
    }

    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    const DepthConfig & c = m.cfg.dd;
    const int nb = (int) items.size();

    if (nb > n_branch) {
        kv.free();
        init(m, std::max(nb, n_branch * 2));
    }
    kv.reset();

    // Allocate single unified forward graph for all 15 depth steps (~2,500 nodes)
    Graph g(16384);

    // Initial inputs: cb0 and hiddens for step 1
    std::vector<int32_t> cb0_idx(nb);
    std::vector<float> flat_hiddens;
    flat_hiddens.reserve((size_t) nb * m.cfg.hidden_size);

    for (int i = 0; i < nb; i++) {
        cb0_idx[i] = items[i].cb0;
        flat_hiddens.insert(flat_hiddens.end(), items[i].hidden_c.begin(), items[i].hidden_c.end());
    }

    ggml_tensor * aud_cb0 = g.input_i32(cb0_idx, nb);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), aud_cb0); // [2048, nb]
    ggml_tensor * h0 = g.input_f32(flat_hiddens, m.cfg.hidden_size, nb);
    embed = ggml_concat(g.ctx, h0, embed, 1); // [2048, 2*nb]

    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, 2*nb]

    std::vector<int32_t> pos_i1(2 * nb);
    for (int i = 0; i < 2 * nb; i++) pos_i1[i] = i / nb;
    ggml_tensor * pos1 = g.input_i32(pos_i1, 2 * nb);
    ggml_tensor * ff = g.input_f32(freq_factors, (int) freq_factors.size());
    std::vector<float> mask_v1 = build_branch_causal_mask(2 * nb, 2 * nb, 0, nb);
    ggml_tensor * mask1 = g.input_f32(mask_v1, 2 * nb, 2 * nb);

    for (int il = 0; il < c.n_layer; il++) {
        x = dd_layer(g.ctx, m, g, kv, x, il, pos1, ff, mask1, 0, 2 * nb);
    }
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, nb, x->nb[1],
                                                       (size_t) nb * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw0 = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1], 0);
    ggml_tensor * logits1 = ggml_mul_mat(g.ctx, hw0, last); // [vocab, nb]

    // Step 1: On-GPU Argmax (Zero PCIe Copy!)
    ggml_tensor * sampled1 = ggml_argmax(g.ctx, logits1); // [nb], GGML_TYPE_I32
    ggml_tensor * all_codes = ggml_reshape_2d(g.ctx, sampled1, nb, 1); // [nb, 1]

    ggml_tensor * prev_sampled = sampled1;

    // Steps 2 to 15: Chained continuously on the GPU with zero CPU round-trips
    for (int j = 2; j < nc; j++) {
        const int prev_cb = j - 1; // codebook from previous step
        const int cur_head = j - 1; // head for current step
        const int start = j;
        const int total = (start + 1) * nb;

        // In-VRAM lookup: view audio_embd.weight for codebook prev_cb
        size_t emb_offset = (size_t) (prev_cb * vs) * m.w("audio_embd.weight")->nb[1];
        ggml_tensor * emb_table = ggml_view_2d(g.ctx, m.w("audio_embd.weight"), m.cfg.hidden_size, vs,
                                               m.w("audio_embd.weight")->nb[1], emb_offset);
        ggml_tensor * aud_j = ggml_get_rows(g.ctx, emb_table, prev_sampled); // [2048, nb]
        ggml_tensor * x_j = linear(g.ctx, m.w("dd.in_proj.weight"), aud_j); // [1024, nb]

        std::vector<int32_t> pos_ij(nb, j);
        ggml_tensor * pos_j = g.input_i32(pos_ij, nb);
        std::vector<float> mask_vj = build_branch_causal_mask(nb, total, start, nb);
        ggml_tensor * mask_j = g.input_f32(mask_vj, total, nb);

        for (int il = 0; il < c.n_layer; il++) {
            x_j = dd_layer(g.ctx, m, g, kv, x_j, il, pos_j, ff, mask_j, start * nb, nb);
        }
        x_j = rms_norm(g.ctx, x_j, m.w("dd.output_norm.weight"), c.rms_eps);

        ggml_tensor * hw_j = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1],
                                          (size_t) cur_head * head->nb[2]);
        ggml_tensor * logits_j = ggml_mul_mat(g.ctx, hw_j, x_j); // [vocab, nb]

        // On-GPU Argmax
        ggml_tensor * sampled_j = ggml_argmax(g.ctx, logits_j); // [nb]
        ggml_tensor * s_2d = ggml_reshape_2d(g.ctx, sampled_j, nb, 1);
        all_codes = ggml_concat(g.ctx, all_codes, s_2d, 1); // shape becomes [nb, j]

        prev_sampled = sampled_j;
    }

    // Single uninterrupted GPU execution for all 15 codebooks!
    g.compute(m.backend, all_codes);

    // Read back the final 15 codebooks in one single contiguous 2.4 KB transfer
    std::vector<int32_t> flat_codes((size_t) nb * (nc - 1));
    ggml_backend_tensor_get(all_codes, flat_codes.data(), 0, flat_codes.size() * sizeof(int32_t));

    std::vector<std::vector<int>> results(nb, std::vector<int>(nc - 1));
    for (int k = 0; k < nc - 1; k++) {
        for (int i = 0; i < nb; i++) {
            results[i][k] = flat_codes[(size_t) k * nb + i];
        }
    }

    return results;
}

}

