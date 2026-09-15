#include "breeze/breeze.h"
#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/backbone.h"
#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"
#include "breeze/text_encoder.h"
#include "breeze/codec.h"

#include <exception>
#include <random>
#include <string>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <algorithm>

using namespace breeze;

struct breeze_context {
    BreezeModel model;
    MimiCodec codec;
};

static std::string g_error;

static GenRequest to_req(const breeze_request * r) {
    GenRequest g;
    g.text = r->text ? r->text : "";
    g.instruction = r->instruction ? r->instruction : "Speak clearly and naturally.";
    g.ref_text = r->ref_text ? r->ref_text : "";
    if (r->ref_audio && r->ref_audio_len > 0)
        g.ref_audio.assign(r->ref_audio, r->ref_audio + r->ref_audio_len);
    g.cfg_scale = r->cfg_scale > 0 ? r->cfg_scale : 1.0f;
    g.seed = r->seed;
    g.max_new_tokens = r->max_new_tokens;
    if (r->split_chars != 0) g.split_chars = r->split_chars < 0 ? 0 : r->split_chars;
    g.temperature = r->temperature;
    g.top_k = r->top_k;
    g.top_p = r->top_p;
    g.repetition_penalty = r->repetition_penalty;
    return g;
}

extern "C" {

breeze_context * breeze_init(const char * gguf_path, int use_gpu) {
    breeze_context * c = new breeze_context();
    try {
        if (!c->model.load(gguf_path, use_gpu != 0)) {
            g_error = "failed to load model";
            delete c;
            return nullptr;
        }
    } catch (const std::exception & e) {
        g_error = e.what();
        delete c;
        return nullptr;
    }
    c->codec.init(c->model);
    return c;
}

void breeze_free(breeze_context * ctx) {
    if (ctx) {
        ctx->model.free();
        delete ctx;
    }
}

int breeze_sample_rate(breeze_context * ctx) {
    return ctx ? ctx->model.cfg.sample_rate : 0;
}

int breeze_generate(breeze_context * ctx, const breeze_request * req, breeze_audio_cb cb, void * user) {
    try {
        GenRequest g = to_req(req);
        generate(ctx->model, ctx->codec, g, [&](const float * s, int n) {
            return cb ? cb(s, n, user) == 0 : true;
        });
    } catch (const std::exception & e) {
        g_error = e.what();
        return 1;
    }
    return 0;
}

int breeze_generate_wav(breeze_context * ctx, const breeze_request * req, const char * out_path) {
    try {
        GenRequest g = to_req(req);
        std::vector<float> audio;
        generate(ctx->model, ctx->codec, g, [&](const float * s, int n) {
            audio.insert(audio.end(), s, s + n);
            return true;
        });
        if (!write_wav(out_path, audio, ctx->model.cfg.sample_rate)) {
            g_error = "failed to write wav";
            return 1;
        }
    } catch (const std::exception & e) {
        g_error = e.what();
        return 1;
    }
    return 0;
}

const char * breeze_last_error(void) {
    return g_error.c_str();
}

} // extern "C"


// --- Multi-Session Generator Slot Definition ---
struct BreezeSession {
    int session_id = 0;
    breeze::BackboneState st_c;
    breeze::BackboneState st_u;
    bool use_cfg = false;
    float cfg_scale = 1.0f;
    std::vector<int> hist;
    std::vector<float> last_hidden;
    std::vector<float> last_hidden_u;
    int last_cb0 = -1;
    std::mt19937 rng;
    breeze::SampleParams bp;
    std::vector<int> suppress;
    bool active = false;
    bool use_q4 = false;
    int steps_taken = 0;
    int max_tokens = 2048;
    bool st_c_init = false;
    bool st_u_init = false;

    void free() {
        if (st_c_init) st_c.free();
        if (st_u_init) st_u.free();
        st_c_init = false;
        st_u_init = false;
        active = false;
        use_q4 = false;
    }
};

static std::vector<float> combine_logits(const std::vector<float> & cond, const std::vector<float> & unc,
                                         bool use_cfg, float scale) {
    if (!use_cfg) return cond;
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < cond.size(); i++)
        out[i] = unc[i] + scale * (cond[i] - unc[i]);
    return out;
}

// --- Dual-Instance Generator & Dedicated Streaming Vocoder Implementation ---
struct breeze_generator {
    breeze::BreezeModel model;
    breeze::DepthRunner depth_single;
    breeze::DepthRunner depth_dual;
    breeze::DepthRunner depth_batched;
    int device = 0;
    bool depth_single_init = false;
    bool depth_dual_init = false;
    bool depth_batched_init = false;

    // Modular INT4 Depth Decoder piece (~300 MiB)
    breeze::GGUFModel q4_dd;
    breeze::BreezeModel model_q4;
    breeze::DepthRunner depth_single_q4;
    breeze::DepthRunner depth_dual_q4;
    breeze::DepthRunner depth_batched_q4;
    bool has_q4_dd = false;
    bool depth_single_q4_init = false;
    bool depth_dual_q4_init = false;
    bool depth_batched_q4_init = false;

    std::unordered_map<int, std::unique_ptr<BreezeSession>> sessions;

    // Legacy single-session compatibility fields
    breeze::BackboneState st;
    std::mt19937 rng;
    std::vector<int> hist;
    std::vector<int> suppress;
    breeze::SampleParams bp;
    std::vector<float> last_hidden;
    bool st_init = false;
};

struct breeze_vocoder {
    breeze::BreezeModel model;
    breeze::MimiCodec codec;
    int device = 1;
    bool codec_init = false;
};

extern "C" {

breeze_generator * breeze_generator_init(const char * gguf_path, int cuda_device) {
    breeze_generator * gen = new breeze_generator();
    gen->device = cuda_device;
    try {
        if (!gen->model.load_device(gguf_path, cuda_device)) {
            g_error = "failed to load model for generator";
            delete gen;
            return nullptr;
        }
        gen->depth_single.init(gen->model, 1);
        gen->depth_single_init = true;
        gen->depth_dual.init(gen->model, 2);
        gen->depth_dual_init = true;
        gen->depth_batched.init(gen->model, 64);
        gen->depth_batched_init = true;
        return gen;
    } catch (const std::exception & e) {
        g_error = e.what();
        delete gen;
        return nullptr;
    }
}

void breeze_generator_free(breeze_generator * gen) {
    if (!gen) return;
    for (auto & pair : gen->sessions) {
        if (pair.second) pair.second->free();
    }
    gen->sessions.clear();
    if (gen->st_init) gen->st.free();
    if (gen->depth_single_init) gen->depth_single.free();
    if (gen->depth_dual_init) gen->depth_dual.free();
    if (gen->depth_batched_init) gen->depth_batched.free();
    if (gen->depth_single_q4_init) gen->depth_single_q4.free();
    if (gen->depth_dual_q4_init) gen->depth_dual_q4.free();
    if (gen->depth_batched_q4_init) gen->depth_batched_q4.free();
    if (gen->has_q4_dd) gen->q4_dd.free();
    gen->model.free();
    delete gen;
}


int breeze_generator_prefill(breeze_generator * gen, const char * text, 
                             const char * instruction, unsigned int seed, int * out_cb0) {
    if (!gen || !text || !out_cb0) return -1;
    try {
        gen->rng.seed(seed);
        gen->bp.temperature = 0.9f;
        gen->bp.top_k = 50;
        gen->bp.top_p = 1.0f;
        gen->bp.repetition_penalty = 1.1f;
        gen->suppress.clear();
        for (int t = gen->model.cfg.codec_codebook_size; t < gen->model.cfg.audio_vocab_size; t++) {
            gen->suppress.push_back(t);
        }
        gen->hist.clear();

        const std::string spk = "[S0]";
        std::string ins = instruction ? instruction : "Speak clearly and naturally.";
        std::string tail = spk + "<ins_bos>" + ins + "<ins_eos>" + text;
        std::vector<int> tokens = gen->model.tok.encode(tail, true);
        std::vector<float> emb_c = breeze::text_encoder_forward(gen->model, tokens);
        int total_c = (int) tokens.size();

        const int max_new = gen->model.cfg.max_new_tokens > 0 ? gen->model.cfg.max_new_tokens : 2048;
        if (gen->st_init) gen->st.free();
        gen->st.init(gen->model, total_c + max_new + 8);
        gen->st_init = true;

        breeze::StepOut o_pref = breeze::backbone_run(gen->model, gen->st, emb_c, total_c);
        int cb0 = breeze::sample_token(o_pref.logits, gen->bp, gen->rng, &gen->hist, &gen->suppress);
        gen->hist.push_back(cb0);
        *out_cb0 = cb0;
        gen->last_hidden = o_pref.hidden;
        return 0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

int breeze_generator_step_frame(breeze_generator * gen, int cb0, 
                                unsigned int seed, int * out_frame_16) {
    if (!gen || !out_frame_16 || !gen->st_init) return -1;
    try {
        // Step 1: 15 autoregressive forward passes through Depth Decoder on this local GPU
        std::mt19937 depth_rng(seed);
        std::vector<std::vector<float>> hiddens = { gen->last_hidden };
        std::vector<int> depth_codes = gen->depth_single.run(gen->model, hiddens, cb0, 1.0f, depth_rng);

        // Step 2: Assemble full 16-codebook frame
        out_frame_16[0] = cb0;
        for (size_t i = 0; i < depth_codes.size() && i < 15; i++) {
            out_frame_16[i + 1] = depth_codes[i];
        }

        // Step 3: Run 1 forward pass through Backbone on this local GPU -> outputs Codebook 0 for next frame
        std::vector<int> frame(out_frame_16, out_frame_16 + 16);
        std::vector<float> ae = breeze::audio_embed_forward(gen->model, frame, 1);
        breeze::StepOut o_c = breeze::backbone_run(gen->model, gen->st, ae, 1);
        int next_cb0 = breeze::sample_token(o_c.logits, gen->bp, gen->rng, &gen->hist, &gen->suppress);
        if (next_cb0 == gen->model.cfg.backbone_eos_token_id) {
            return -1; // EOS reached
        }
        gen->hist.push_back(next_cb0);
        gen->last_hidden = o_c.hidden;
        return next_cb0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

// --- Multi-Session Dynamic Batching & Interleaved Round-Robin Implementation ---
BREEZE_API int breeze_generator_session_create_ext(breeze_generator * gen, int session_id, 
                                                   const char * text, const char * instruction,
                                                   const char * ref_text, const int * ref_codes, int ref_frames,
                                                   float cfg_scale, unsigned int seed, int max_new_tokens,
                                                   int * out_cb0, int * out_allocated_tokens) {
    if (!gen || !text || !out_cb0) return -1;
    try {
        // Free existing slot if session_id is being reused
        auto it = gen->sessions.find(session_id);
        if (it != gen->sessions.end()) {
            if (it->second) it->second->free();
            gen->sessions.erase(it);
        }

        auto sess = std::make_unique<BreezeSession>();
        sess->session_id = session_id;
        sess->cfg_scale = cfg_scale > 0.0f ? cfg_scale : 1.0f;
        sess->use_cfg = (sess->cfg_scale != 1.0f);
        sess->rng.seed(seed);
        sess->bp.temperature = 0.9f;
        sess->bp.top_k = 50;
        sess->bp.top_p = 1.0f;
        sess->bp.repetition_penalty = 1.1f;
        sess->suppress.clear();
        for (int t = gen->model.cfg.codec_codebook_size; t < gen->model.cfg.audio_vocab_size; t++) {
            sess->suppress.push_back(t);
        }
        sess->hist.clear();

        // 1. Dynamic sweet-spot estimation if not explicitly provided
        int est_out_frames = max_new_tokens;
        if (est_out_frames <= 0) {
            int words = 0;
            bool in_w = false;
            for (const char * p = text; *p; ++p) {
                if (std::isspace((unsigned char)*p)) in_w = false;
                else if (!in_w) { in_w = true; words++; }
            }
            est_out_frames = std::min(1000, (int)std::ceil(words * 2.2f) + 60);
        }

        const std::string spk = "[S0]";
        std::string ins = instruction ? instruction : "Speak clearly and naturally.";
        bool has_ref = (ref_codes != nullptr && ref_frames > 0 && ref_text != nullptr && strlen(ref_text) > 0);

        auto add_text_seg = [&](const std::string & str, std::vector<float> & out_emb, int & total) {
            std::vector<int> toks = gen->model.tok.encode(str, true);
            std::vector<float> e = breeze::text_encoder_forward(gen->model, toks);
            out_emb.insert(out_emb.end(), e.begin(), e.end());
            total += (int) toks.size();
        };

        auto add_audio_seg = [&](const int * codes, int n_frames, std::vector<float> & out_emb, int & total) {
            std::vector<int> vcodes(codes, codes + (size_t) n_frames * gen->model.cfg.num_codebooks);
            std::vector<float> e = breeze::audio_embed_forward(gen->model, vcodes, n_frames);
            out_emb.insert(out_emb.end(), e.begin(), e.end());
            total += n_frames;
            std::vector<int> eos_frame(gen->model.cfg.num_codebooks, gen->model.cfg.codebook_eos_token_id);
            std::vector<float> ee = breeze::audio_embed_forward(gen->model, eos_frame, 1);
            out_emb.insert(out_emb.end(), ee.begin(), ee.end());
            total += 1;
        };
        
        // 2. Conditional prefill
        int total_c = 0;
        std::vector<float> emb_c;
        if (has_ref) {
            add_text_seg(spk + ref_text, emb_c, total_c);
            add_audio_seg(ref_codes, ref_frames, emb_c, total_c);
        }
        std::string tail_c = spk + "<ins_bos>" + ins + "<ins_eos>" + text;
        add_text_seg(tail_c, emb_c, total_c);

        // Clamp to strictly guarantee total sequence <= 2040 (2048 backbone context limit)
        int max_available = 2040 - total_c;
        if (est_out_frames > max_available) {
            est_out_frames = std::max(64, max_available);
        }
        sess->max_tokens = est_out_frames;

        const int alloc_c = total_c + sess->max_tokens + 8;
        sess->st_c.init(gen->model, alloc_c);
        sess->st_c_init = true;
        breeze::StepOut o_c = breeze::backbone_run(gen->model, sess->st_c, emb_c, total_c);
        sess->last_hidden = o_c.hidden;

        // 3. Unconditional prefill (if CFG > 1.0)
        breeze::StepOut o_u;
        int alloc_u = 0;
        if (sess->use_cfg) {
            int total_u = 0;
            std::vector<float> emb_u;
            if (has_ref) {
                add_text_seg(spk + ref_text, emb_u, total_u);
                add_audio_seg(ref_codes, ref_frames, emb_u, total_u);
            }
            std::string tail_u = spk + text;
            add_text_seg(tail_u, emb_u, total_u);

            alloc_u = total_u + sess->max_tokens + 8;
            sess->st_u.init(gen->model, alloc_u);
            sess->st_u_init = true;
            o_u = breeze::backbone_run(gen->model, sess->st_u, emb_u, total_u);
            sess->last_hidden_u = o_u.hidden;
        }

        // Return exact allocated token capacity to caller if requested
        if (out_allocated_tokens) {
            *out_allocated_tokens = sess->use_cfg ? (alloc_c + alloc_u) : alloc_c;
        }

        // 4. Sample initial cb0
        std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, sess->use_cfg, sess->cfg_scale);
        int cb0 = breeze::sample_token(comb, sess->bp, sess->rng, &sess->hist, &sess->suppress);
        sess->hist.push_back(cb0);
        sess->last_cb0 = cb0;
        sess->active = true;
        sess->steps_taken = 0;

        *out_cb0 = cb0;
        gen->sessions[session_id] = std::move(sess);
        return 0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

int breeze_generator_session_create(breeze_generator * gen, int session_id, 
                                    const char * text, const char * instruction, 
                                    float cfg_scale, unsigned int seed, int * out_cb0) {
    return breeze_generator_session_create_ext(gen, session_id, text, instruction, nullptr, nullptr, 0, cfg_scale, seed, 0, out_cb0, nullptr);
}

int breeze_generator_session_step(breeze_generator * gen, int session_id, 
                                  unsigned int seed, int * out_frame_16) {
    if (!gen || !out_frame_16) return -1;
    auto it = gen->sessions.find(session_id);
    if (it == gen->sessions.end() || !it->second || !it->second->active) return -1;
    BreezeSession * sess = it->second.get();

    try {
        int cb0 = sess->last_cb0;
        std::mt19937 depth_rng(seed);

        // Step 1: 15 depth passes (single or batched dual)
        std::vector<int> depth_codes;
        const bool use_q4 = (sess->use_q4 && gen->has_q4_dd);
        breeze::DepthRunner & dr_dual = use_q4 ? gen->depth_dual_q4 : gen->depth_dual;
        breeze::DepthRunner & dr_single = use_q4 ? gen->depth_single_q4 : gen->depth_single;
        breeze::BreezeModel & active_model = use_q4 ? gen->model_q4 : gen->model;

        if (sess->use_cfg) {
            std::vector<std::vector<float>> hiddens = { sess->last_hidden, sess->last_hidden_u };
            depth_codes = dr_dual.run(active_model, hiddens, cb0, sess->cfg_scale, depth_rng);
        } else {
            std::vector<std::vector<float>> hiddens = { sess->last_hidden };
            depth_codes = dr_single.run(active_model, hiddens, cb0, 1.0f, depth_rng);
        }

        // Step 2: Assemble full 16-codebook frame
        out_frame_16[0] = cb0;
        for (size_t i = 0; i < depth_codes.size() && i < 15; i++) {
            out_frame_16[i + 1] = depth_codes[i];
        }

        // Step 3: Run backbone on this local GPU -> outputs Codebook 0 for next frame
        std::vector<int> frame(out_frame_16, out_frame_16 + 16);
        std::vector<float> ae = breeze::audio_embed_forward(gen->model, frame, 1);
        breeze::StepOut o_c = breeze::backbone_run(gen->model, sess->st_c, ae, 1);
        breeze::StepOut o_u;
        if (sess->use_cfg) {
            o_u = breeze::backbone_run(gen->model, sess->st_u, ae, 1);
        }

        std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, sess->use_cfg, sess->cfg_scale);
        int next_cb0 = breeze::sample_token(comb, sess->bp, sess->rng, &sess->hist, &sess->suppress);
        sess->steps_taken++;

        if (next_cb0 == gen->model.cfg.backbone_eos_token_id || sess->steps_taken >= sess->max_tokens) {
            sess->active = false;
            return -1; // EOS reached
        }

        sess->hist.push_back(next_cb0);
        sess->last_cb0 = next_cb0;
        sess->last_hidden = o_c.hidden;
        if (sess->use_cfg) sess->last_hidden_u = o_u.hidden;

        return next_cb0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

int breeze_generator_session_free(breeze_generator * gen, int session_id) {
    if (!gen) return -1;
    auto it = gen->sessions.find(session_id);
    if (it != gen->sessions.end()) {
        if (it->second) it->second->free();
        gen->sessions.erase(it);
        return 0;
    }
    return -1;
}

int breeze_generator_load_q4_depth(breeze_generator * gen, const char * q4_gguf_path) {
    if (!gen || !q4_gguf_path) return -1;
    try {
        if (gen->has_q4_dd) {
            if (gen->depth_single_q4_init) gen->depth_single_q4.free();
            if (gen->depth_dual_q4_init) gen->depth_dual_q4.free();
            if (gen->depth_batched_q4_init) gen->depth_batched_q4.free();
            gen->q4_dd.free();
            gen->has_q4_dd = false;
            gen->depth_single_q4_init = false;
            gen->depth_dual_q4_init = false;
            gen->depth_batched_q4_init = false;
        }

        if (!gen->q4_dd.load_prefix(q4_gguf_path, gen->model.backend, "dd.")) {
            g_error = "failed to load modular Q4 depth decoder piece";
            return -1;
        }

        gen->model_q4.backend = gen->model.backend;
        gen->model_q4.cfg = gen->model.cfg;
        gen->model_q4.tok = gen->model.tok;
        gen->model_q4.base_model = &gen->model;
        gen->model_q4.dd_override = &gen->q4_dd;

        gen->depth_single_q4.init(gen->model_q4, 1);
        gen->depth_single_q4_init = true;
        gen->depth_dual_q4.init(gen->model_q4, 2);
        gen->depth_dual_q4_init = true;
        gen->depth_batched_q4.init(gen->model_q4, 64);
        gen->depth_batched_q4_init = true;
        gen->has_q4_dd = true;
        return 0;

    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

int breeze_generator_session_set_q4(breeze_generator * gen, int session_id, int use_q4) {
    if (!gen) return -1;
    auto it = gen->sessions.find(session_id);
    if (it == gen->sessions.end() || !it->second) return -1;
    it->second->use_q4 = (use_q4 != 0 && gen->has_q4_dd);
    return 0;
}

int breeze_generator_session_count(breeze_generator * gen) {
    return gen ? (int) gen->sessions.size() : 0;
}

int breeze_generator_sessions_step_round(breeze_generator * gen, 
                                        const int * session_ids, int num_sessions, 
                                        unsigned int seed,
                                        int * out_frames_16, int * out_next_cb0) {
    if (!gen || !session_ids || num_sessions <= 0 || !out_frames_16 || !out_next_cb0) return 0;
    int completed_in_round = 0;
    for (int i = 0; i < num_sessions; i++) {
        int sid = session_ids[i];
        int next_cb = breeze_generator_session_step(gen, sid, seed + i, out_frames_16 + i * 16);
        out_next_cb0[i] = next_cb;
        if (next_cb >= 0) {
            completed_in_round++;
        }
    }
    return completed_in_round;
}

int breeze_generator_sessions_step_batched(breeze_generator * gen, 
                                          const int * session_ids, int num_sessions, 
                                          const unsigned int * seeds,
                                          int * out_frames_16, int * out_next_cb0) {
    if (!gen || !session_ids || num_sessions <= 0 || !out_frames_16 || !out_next_cb0) return 0;
    try {
        // 1. Separate sessions into Q8 and Q4 groups
        std::vector<breeze::DepthRunner::BatchItem> items_q8;
        std::vector<breeze::DepthRunner::BatchItem> items_q4;
        std::vector<int> map_q8_to_i;
        std::vector<int> map_q4_to_i;

        for (int i = 0; i < num_sessions; i++) {
            out_next_cb0[i] = -1;
            int sid = session_ids[i];
            auto it = gen->sessions.find(sid);
            if (it == gen->sessions.end() || !it->second || !it->second->active) {
                continue;
            }
            BreezeSession * sess = it->second.get();

            breeze::DepthRunner::BatchItem item;
            item.session_id = sid;
            item.cb0 = sess->last_cb0;
            item.cfg_scale = sess->cfg_scale;
            item.use_cfg = sess->use_cfg;
            item.hidden_c = sess->last_hidden;
            if (sess->use_cfg) item.hidden_u = sess->last_hidden_u;
            item.rng = &sess->rng;
            if (seeds) {
                sess->rng.seed(seeds[i]);
            }
            item.sp = nullptr;

            if (sess->use_q4 && gen->has_q4_dd) {
                items_q4.push_back(std::move(item));
                map_q4_to_i.push_back(i);
            } else {
                items_q8.push_back(std::move(item));
                map_q8_to_i.push_back(i);
            }
        }

        if (items_q8.empty() && items_q4.empty()) return 0;

        // 2. Run Batched Depth Decoder for each group
        std::vector<std::vector<int>> codes_q8;
        if (!items_q8.empty()) {
            codes_q8 = gen->depth_batched.run_batched(gen->model, items_q8);
        }
        std::vector<std::vector<int>> codes_q4;
        if (!items_q4.empty()) {
            codes_q4 = gen->depth_batched_q4.run_batched(gen->model_q4, items_q4);
        }

        // 3. Assemble 16-codebook frames and run Batched Audio Embedding (1 GPU Call!)
        const int num_active_sessions = (int) (items_q8.size() + items_q4.size());
        std::vector<int> all_frames(num_active_sessions * 16);

        struct ActiveSessionMeta {
            int out_idx;
            int sid;
            BreezeSession * sess;
        };
        std::vector<ActiveSessionMeta> active_metas;
        active_metas.reserve(num_active_sessions);

        auto collect_frame = [&](int i, int sid, const std::vector<int> & codes) {
            BreezeSession * sess = gen->sessions[sid].get();
            int * frame_ptr = out_frames_16 + i * 16;
            frame_ptr[0] = sess->last_cb0;
            for (size_t k = 0; k < codes.size() && k < 15; k++) {
                frame_ptr[k + 1] = codes[k];
            }
            int ord = (int) active_metas.size();
            std::memcpy(all_frames.data() + ord * 16, frame_ptr, 16 * sizeof(int));
            active_metas.push_back({ i, sid, sess });
        };

        for (size_t k = 0; k < items_q8.size(); k++) {
            collect_frame(map_q8_to_i[k], items_q8[k].session_id, codes_q8[k]);
        }
        for (size_t k = 0; k < items_q4.size(); k++) {
            collect_frame(map_q4_to_i[k], items_q4[k].session_id, codes_q4[k]);
        }

        // Single batched GPU lookup for all sessions simultaneously!
        std::vector<float> all_ae = breeze::audio_embed_forward(gen->model, all_frames, num_active_sessions);
        const int hidden_size = gen->model.cfg.hidden_size;

        struct SessionBackboneMapping {
            int out_idx;
            int sid;
            int bb_idx_c;
            int bb_idx_u;
        };
        std::vector<SessionBackboneMapping> bb_map;
        bb_map.reserve(num_active_sessions);
        std::vector<breeze::BackboneBatchItem> bb_items;
        bb_items.reserve(num_active_sessions * 2);

        for (int m = 0; m < num_active_sessions; m++) {
            const auto & meta = active_metas[m];
            const float * ae_ptr = all_ae.data() + (size_t) m * hidden_size;
            std::vector<float> ae(ae_ptr, ae_ptr + hidden_size);

            SessionBackboneMapping mapping;
            mapping.out_idx = meta.out_idx;
            mapping.sid = meta.sid;

            // Conditional branch
            mapping.bb_idx_c = (int) bb_items.size();
            bb_items.push_back({ &meta.sess->st_c, ae });

            // Unconditional branch (if CFG > 1.0)
            if (meta.sess->use_cfg) {
                mapping.bb_idx_u = (int) bb_items.size();
                bb_items.push_back({ &meta.sess->st_u, ae });
            } else {
                mapping.bb_idx_u = -1;
            }

            bb_map.push_back(mapping);
        }

        // 4. Run Batched Backbone GEMM across ALL sessions simultaneously!
        std::vector<breeze::StepOut> bb_outs = breeze::backbone_run_batched(gen->model, bb_items);

        // 5. Unpack Backbone outputs, sample next cb0, check EOS
        int completed_in_round = 0;
        for (const auto & m : bb_map) {
            BreezeSession * sess = gen->sessions[m.sid].get();
            const breeze::StepOut & o_c = bb_outs[m.bb_idx_c];
            breeze::StepOut o_u;
            if (sess->use_cfg && m.bb_idx_u >= 0) {
                o_u = bb_outs[m.bb_idx_u];
            }

            std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, sess->use_cfg, sess->cfg_scale);
            int next_cb0 = breeze::sample_token(comb, sess->bp, sess->rng, &sess->hist, &sess->suppress);
            sess->steps_taken++;

            if (next_cb0 == gen->model.cfg.backbone_eos_token_id || sess->steps_taken >= sess->max_tokens) {
                sess->active = false;
                out_next_cb0[m.out_idx] = -1;
            } else {
                sess->hist.push_back(next_cb0);
                sess->last_cb0 = next_cb0;
                sess->last_hidden = o_c.hidden;
                if (sess->use_cfg) sess->last_hidden_u = o_u.hidden;
                out_next_cb0[m.out_idx] = next_cb0;
                completed_in_round++;
            }
        }

        return completed_in_round;

    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}


breeze_vocoder * breeze_vocoder_init(const char * gguf_path, int cuda_device) {
    breeze_vocoder * voc = new breeze_vocoder();
    voc->device = cuda_device;
    try {
        if (!voc->model.load_device(gguf_path, cuda_device)) {
            g_error = "failed to load model for vocoder";
            delete voc;
            return nullptr;
        }
        voc->codec.init(voc->model);
        voc->codec_init = true;
        return voc;
    } catch (const std::exception & e) {
        g_error = e.what();
        delete voc;
        return nullptr;
    }
}

void breeze_vocoder_free(breeze_vocoder * voc) {
    if (!voc) return;
    voc->model.free();
    delete voc;
}

int breeze_vocoder_stream_decode(breeze_vocoder * voc, const int * frames, 
                                 int n_frames, float * out_pcm) {
    if (!voc || !frames || n_frames <= 0 || !out_pcm) return 0;
    try {
        const int nc = voc->model.cfg.num_codebooks;
        std::vector<int> sub(frames, frames + n_frames * nc);
        std::vector<float> audio = voc->codec.decode(sub, n_frames);
        std::memcpy(out_pcm, audio.data(), audio.size() * sizeof(float));
        return (int) audio.size();
    } catch (const std::exception & e) {
        g_error = e.what();
        return 0;
    }
}

} // extern "C"
