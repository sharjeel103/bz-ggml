#ifndef BREEZE_H
#define BREEZE_H

#include <stddef.h>

#if defined(_WIN32)
#ifdef BREEZE_BUILD_SHARED
#define BREEZE_API __declspec(dllexport)
#else
#define BREEZE_API
#endif
#else
#define BREEZE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct breeze_context breeze_context;

// return 0 to continue, non-zero to stop generation
typedef int (*breeze_audio_cb)(const float * samples, int n_samples, void * user);

typedef struct {
    const char * text;
    const char * instruction;   // null for a neutral default
    const char * ref_text;      // null when not cloning
    const float * ref_audio;    // mono 24 kHz samples, null for voice design
    int ref_audio_len;
    float cfg_scale;
    int seed;
    int max_new_tokens;         // 0 uses the model default
    int split_chars;            // 0 uses the default, negative keeps long text in a single pass
    // sampling. zero on any of these keeps whatever the gguf was built with
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
} breeze_request;

BREEZE_API breeze_context * breeze_init(const char * gguf_path, int use_gpu);
BREEZE_API void breeze_free(breeze_context * ctx);
BREEZE_API int breeze_sample_rate(breeze_context * ctx);

// streams audio chunks to the callback; returns 0 on success
BREEZE_API int breeze_generate(breeze_context * ctx, const breeze_request * req,
                               breeze_audio_cb cb, void * user);

// convenience: generate and write a 16-bit PCM WAV; returns 0 on success
BREEZE_API int breeze_generate_wav(breeze_context * ctx, const breeze_request * req,
                                   const char * out_path);

BREEZE_API const char * breeze_last_error(void);

// --- Dual-Instance Generator & Dedicated Streaming Vocoder C API ---
typedef struct breeze_generator breeze_generator;
typedef struct breeze_vocoder breeze_vocoder;

// Generator: Co-located Backbone + Depth Decoder on a dedicated CUDA device
BREEZE_API breeze_generator * breeze_generator_init(const char * gguf_path, int cuda_device);
BREEZE_API void breeze_generator_free(breeze_generator * gen);

// Prefill text prompt and initialize generator's local KV cache
BREEZE_API int breeze_generator_prefill(breeze_generator * gen, const char * text, 
                                        const char * instruction, unsigned int seed, 
                                        int * out_cb0);

// Execute one complete frame step: 15 depth passes -> assemble 16 codebooks -> 1 backbone pass
// Returns next cb0 (or -1 if EOS reached)
BREEZE_API int breeze_generator_step_frame(breeze_generator * gen, int cb0, 
                                           unsigned int seed, int * out_frame_16);

// Dedicated Neural Vocoder on specified CUDA device
BREEZE_API breeze_vocoder * breeze_vocoder_init(const char * gguf_path, int cuda_device);
BREEZE_API void breeze_vocoder_free(breeze_vocoder * voc);

// Stream decode frame chunk (1 to N frames) into 24 kHz float PCM
BREEZE_API int breeze_vocoder_stream_decode(breeze_vocoder * voc, const int * frames, 
                                            int n_frames, float * out_pcm);

#ifdef __cplusplus
}
#endif

#endif
