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


// --- Dual-Instance Generator & Dedicated Streaming Vocoder Implementation ---
struct breeze_generator {
    breeze::BreezeModel model;
    breeze::BackboneState st;
    breeze::DepthRunner depth;
    std::mt19937 rng;
    std::vector<int> hist;
    std::vector<int> suppress;
    breeze::SampleParams bp;
    std::vector<float> last_hidden;
    int device = 0;
    bool st_init = false;
    bool depth_init = false;
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
        gen->depth.init(gen->model, 1);
        gen->depth_init = true;
        return gen;
    } catch (const std::exception & e) {
        g_error = e.what();
        delete gen;
        return nullptr;
    }
}

void breeze_generator_free(breeze_generator * gen) {
    if (!gen) return;
    if (gen->st_init) gen->st.free();
    if (gen->depth_init) gen->depth.free();
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
        std::vector<int> depth_codes = gen->depth.run(gen->model, hiddens, cb0, 1.0f, depth_rng);

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
