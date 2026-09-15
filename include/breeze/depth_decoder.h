#pragma once

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <random>
#include <vector>

namespace breeze {

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;

    void init(BreezeModel & m, int n_branches);
    void free();

    // hiddens holds the backbone last hidden per branch (cond first, then uncond); returns cb1..cb_{n-1}.
    // sp overrides the model's own sampling settings, null uses them.
    // force supplies the first n_force codebooks instead of sampling them
    std::vector<int> run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                         int cb0, float cfg_scale, std::mt19937 & rng,
                         const SampleParams * sp = nullptr,
                         const int * force = nullptr, int n_force = 0);

    // Multi-session batched execution: runs all 15 depth steps simultaneously in batched GEMM
    struct BatchItem {
        int session_id = 0;
        int cb0 = 0;
        float cfg_scale = 1.0f;
        bool use_cfg = false;
        std::vector<float> hidden_c;
        std::vector<float> hidden_u;
        std::mt19937 * rng = nullptr;
        const SampleParams * sp = nullptr;
    };

    std::vector<std::vector<int>> run_batched(BreezeModel & m, const std::vector<BatchItem> & items);

    // Unified 15-step on-GPU unrolled graph: executes all 15 codebook decodes in a single GPU pass
    // using On-GPU Argmax and in-VRAM embedding lookups, completely eliminating PCIe round-trips.
    std::vector<std::vector<int>> run_batched_unified_gpu(BreezeModel & m, const std::vector<BatchItem> & items);
};

}

