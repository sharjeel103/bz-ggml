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
    int device = 0;
    bool depth_single_init = false;
    bool depth_dual_init = false;

    // Modular INT4 Depth Decoder piece (~300 MiB)
    breeze::GGUFModel q4_dd;
    breeze::BreezeModel model_q4;
    breeze::DepthRunner depth_single_q4;
    breeze::DepthRunner depth_dual_q4;
    bool has_q4_dd = false;
    bool depth_single_q4_init = false;
    bool depth_dual_q4_init = false;

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
    if (gen->depth_single_q4_init) gen->depth_single_q4.free();
    if (gen->depth_dual_q4_init) gen->depth_dual_q4.free();
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
int breeze_generator_session_create(breeze_generator * gen, int session_id, 
                                    const char * text, const char * instruction, 
                                    float cfg_scale, unsigned int seed, int * out_cb0) {
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
        sess->max_tokens = gen->model.cfg.max_new_tokens > 0 ? gen->model.cfg.max_new_tokens : 2048;

        const std::string spk = "[S0]";
        std::string ins = instruction ? instruction : "Speak clearly and naturally.";
        
        // 1. Conditional prefill
        std::string tail_c = spk + "<ins_bos>" + ins + "<ins_eos>" + text;
        std::vector<int> tokens_c = gen->model.tok.encode(tail_c, true);
        std::vector<float> emb_c = breeze::text_encoder_forward(gen->model, tokens_c);
        int total_c = (int) tokens_c.size();

        sess->st_c.init(gen->model, total_c + sess->max_tokens + 8);
        sess->st_c_init = true;
        breeze::StepOut o_c = breeze::backbone_run(gen->model, sess->st_c, emb_c, total_c);
        sess->last_hidden = o_c.hidden;

        // 2. Unconditional prefill (if CFG > 1.0)
        breeze::StepOut o_u;
        if (sess->use_cfg) {
            std::string tail_u = spk + text;
            std::vector<int> tokens_u = gen->model.tok.encode(tail_u, true);
            std::vector<float> emb_u = breeze::text_encoder_forward(gen->model, tokens_u);
            int total_u = (int) tokens_u.size();

            sess->st_u.init(gen->model, total_u + sess->max_tokens + 8);
            sess->st_u_init = true;
            o_u = breeze::backbone_run(gen->model, sess->st_u, emb_u, total_u);
            sess->last_hidden_u = o_u.hidden;
        }

        // 3. Sample initial cb0
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
            gen->q4_dd.free();
            gen->has_q4_dd = false;
            gen->depth_single_q4_init = false;
            gen->depth_dual_q4_init = false;
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
