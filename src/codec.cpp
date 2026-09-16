#include "breeze/codec.h"

#include <cfloat>
#include <cmath>
#include <string>

namespace breeze {

using namespace codec_detail;

static ggml_tensor * transpose_cont(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

std::vector<float> MimiCodec::decode(const std::vector<int> & codes, int T, int n_cb) {
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    Graph g(32768);
    ggml_tensor * x = vocoder_decode(g.ctx, *m, g, codes, n_cb, T);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    g.compute(m->backend, audio);
    return tensor_to_f32(audio);
}

std::vector<std::vector<float>> MimiCodec::decode_batch(
    const std::vector<std::vector<int>> & batch_codes,
    const std::vector<int> & batch_T,
    int n_cb)
{
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    int B = (int)batch_codes.size();
    if (B <= 0) return {};
    if (B == 1) {
        return { decode(batch_codes[0], batch_T[0], n_cb) };
    }

    size_t graph_size = std::max((size_t)65536, (size_t)B * 2048);
    Graph g(graph_size);

    std::vector<ggml_tensor *> audio_nodes(B);
    for (int b = 0; b < B; b++) {
        ggml_tensor * x = vocoder_decode(g.ctx, *m, g, batch_codes[b], n_cb, batch_T[b]);
        ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
        ggml_set_output(audio);
        audio_nodes[b] = audio;
        if (b < B - 1) {
            g.write(audio);
        }
    }

    g.compute(m->backend, audio_nodes.back());

    std::vector<std::vector<float>> results(B);
    for (int b = 0; b < B; b++) {
        results[b] = tensor_to_f32(audio_nodes[b]);
    }
    return results;
}

static int nearest(const std::vector<float> & book, const float * v, int dim, int n) {
    int best = 0;
    float best_d = FLT_MAX;
    for (int c = 0; c < n; c++) {
        const float * b = &book[(size_t) c * dim];
        float d = 0.0f;
        for (int k = 0; k < dim; k++) {
            float diff = v[k] - b[k];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

static std::vector<float> read_book(BreezeModel & m, const std::string & name) {
    return tensor_to_f32(m.w(name));
}

std::vector<int> MimiCodec::encode(const std::vector<float> & audio, int & n_frames) {
    const int nc = m->cfg.num_codebooks;
    const int n_sem = m->cfg.codec.num_semantic;
    const int n_ac = nc - n_sem;
    const int dim = m->cfg.codec.codebook_dim;
    const int book = m->cfg.codec_codebook_size;

    Graph g(16384);
    ggml_tensor * x = g.input_f32(audio, (int) audio.size(), 1); // [L, 1]
    x = seanet_encoder(g.ctx, *m, x);                            // [T, hidden]
    x = transpose_cont(g.ctx, x);                                // [hidden, T]
    x = mimi_transformer(g.ctx, *m, g, x, "codec.enct", (int) x->ne[1]);
    x = transpose_cont(g.ctx, x);                                // [T, hidden]
    x = conv1d_causal(g.ctx, m->w("codec.downsample.conv.weight"), nullptr, x, 2, 1); // [T2, hidden]
    ggml_tensor * emb = transpose_cont(g.ctx, x);                // [hidden, T2]
    ggml_tensor * rs = linear(g.ctx, m->w("codec.sq.in_proj.weight"), emb); // [dim, T2]
    ggml_tensor * ra = linear(g.ctx, m->w("codec.aq.in_proj.weight"), emb); // [dim, T2]
    ggml_set_output(rs);
    g.write(rs);
    g.compute(m->backend, ra);

    std::vector<float> rs_h = tensor_to_f32(rs);
    std::vector<float> ra_h = tensor_to_f32(ra);
    const int T = (int) ra->ne[1];
    n_frames = T;

    std::vector<float> sbook = read_book(*m, "codec.sq.0.embed");
    std::vector<std::vector<float>> abook(n_ac);
    for (int i = 0; i < n_ac; i++) abook[i] = read_book(*m, "codec.aq." + std::to_string(i) + ".embed");

    std::vector<int> codes((size_t) T * nc);
    std::vector<float> res(dim);
    for (int f = 0; f < T; f++) {
        codes[(size_t) f * nc + 0] = nearest(sbook, &rs_h[(size_t) f * dim], dim, book);
        for (int k = 0; k < dim; k++) res[k] = ra_h[(size_t) f * dim + k];
        for (int i = 0; i < n_ac; i++) {
            int c = nearest(abook[i], res.data(), dim, book);
            codes[(size_t) f * nc + (n_sem + i)] = c;
            const float * b = &abook[i][(size_t) c * dim];
            for (int k = 0; k < dim; k++) res[k] -= b[k];
        }
    }
    return codes;
}

}
