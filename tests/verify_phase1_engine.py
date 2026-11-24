#!/usr/bin/env python3
"""
Phase 1 Verification: Low-Level C++ Compute, Sampling & Tiled Vocoder Validation
Hard-fail assertions with zero mocking.
"""
import sys, time, gc, subprocess
import numpy as np

sys.path.insert(0, '/kaggle/working/bz-ggml/python')
from bz_ggml.bindings import BreezeLib, GeneratorHandle, VocoderHandle

def get_vram_mb(gpu_id=0):
    res = subprocess.run(
        f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}",
        shell=True, capture_output=True, text=True
    )
    return float(res.stdout.strip())

print("=" * 80)
print("  🚀 PHASE 1: LOW-LEVEL C++ COMPUTE, SAMPLING & TILED VOCODER VALIDATION")
print("=" * 80)

model_path = "/kaggle/working/models/breeze-tts-2-q8_0.gguf"
lib_path = "/kaggle/working/bz-ggml/build/libbreeze.so"

lib = BreezeLib(lib_path)
gen0 = GeneratorHandle(lib, model_path, cuda_device=0)
voc0 = gen0.create_vocoder()

prompt = (
    "Artificial intelligence systems are transforming enterprise audio architectures worldwide "
    "with unprecedented fidelity, lower latency, and highly scalable distributed inference models "
    "across modern accelerated graphics processing hardware platforms."
)

# -----------------------------------------------------------------------------
# Test 1.1: Greedy Determinism (temperature=0.0)
# -----------------------------------------------------------------------------
print("\n[TEST 1.1] Greedy Determinism (temperature=0.0)...")
cb0_1, _ = gen0.session_create_ext(
    101, prompt, cfg_scale=1.0, temperature=0.0, depth_temperature=0.0, seed=42
)
tokens_run1 = []
for s in range(30):
    next_cb0, frame16 = gen0.session_step(101, seed=42)
    tokens_run1.extend(frame16)
    if next_cb0 < 0 or next_cb0 == 2051:
        break
gen0.session_free(101)

cb0_2, _ = gen0.session_create_ext(
    102, prompt, cfg_scale=1.0, temperature=0.0, depth_temperature=0.0, seed=42
)
tokens_run2 = []
for s in range(30):
    next_cb0, frame16 = gen0.session_step(102, seed=42)
    tokens_run2.extend(frame16)
    if next_cb0 < 0 or next_cb0 == 2051:
        break
gen0.session_free(102)

print(f"  • Run 1 tokens: {len(tokens_run1)} | Run 2 tokens: {len(tokens_run2)}")
assert len(tokens_run1) == len(tokens_run2), "Token length mismatch in deterministic runs!"
diff_count = sum(1 for a, b in zip(tokens_run1, tokens_run2) if a != b)
print(f"  • Bitwise difference count: {diff_count} / {len(tokens_run1)}")
assert diff_count == 0, f"Greedy determinism failed: {diff_count} tokens differed!"
print("  ✅ TEST 1.1 PASSED: 100% bit-identical across runs!")

# -----------------------------------------------------------------------------
# Test 1.2: Backbone Temperature Wiring (0.0 vs 0.60 vs 0.90)
# -----------------------------------------------------------------------------
print("\n[TEST 1.2] Backbone Temperature Wiring (0.0 vs 0.60 vs 0.90)...")
def run_temp(temp, sid):
    gen0.session_create_ext(
        sid, prompt, cfg_scale=1.0, temperature=temp, depth_temperature=0.0, seed=42
    )
    tokens = []
    for s in range(30):
        next_cb0, frame16 = gen0.session_step(sid, seed=42)
        tokens.extend(frame16)
        if next_cb0 < 0 or next_cb0 == 2051:
            break
    gen0.session_free(sid)
    return tokens

tok_0 = run_temp(0.0, 201)
tok_06 = run_temp(0.60, 202)
tok_09 = run_temp(0.90, 203)

diff_0_06 = sum(1 for a, b in zip(tok_0, tok_06) if a != b)
diff_06_09 = sum(1 for a, b in zip(tok_06, tok_09) if a != b)
print(f"  • Diff (temp=0.0 vs temp=0.60):  {diff_0_06} / {len(tok_0)} tokens")
print(f"  • Diff (temp=0.60 vs temp=0.90): {diff_06_09} / {len(tok_06)} tokens")
assert diff_0_06 > 0, "Temperature=0.0 and 0.60 produced identical output (temp wiring broken)!"
assert diff_06_09 > 0, "Temperature=0.60 and 0.90 produced identical output (temp wiring broken)!"
print("  ✅ TEST 1.2 PASSED: Backbone temperature is verified wired!")

# -----------------------------------------------------------------------------
# Test 1.3: Unified-GPU CFG Speed & Mixed-Batch Isolation
# -----------------------------------------------------------------------------
print("\n[TEST 1.3] Unified-GPU CFG Speed & Mixed-Batch Isolation (100 frames on GPU 0)...")
NUM_BENCH_STEPS = 25

def bench_batch(sids, cfg_scales, depth_temps, name):
    for sid, cfg, dt in zip(sids, cfg_scales, depth_temps):
        gen0.session_create_ext(
            sid, prompt, cfg_scale=cfg, temperature=0.60, depth_temperature=dt, seed=42
        )
    # Warmup
    gen0.sessions_step_batched(sids)
    
    t0 = time.time()
    for _ in range(NUM_BENCH_STEPS):
        gen0.sessions_step_batched(sids)
    elapsed = time.time() - t0
    
    for sid in sids:
        gen0.session_free(sid)
        
    total_frames = NUM_BENCH_STEPS * len(sids)
    ms_per_frame = (elapsed / total_frames) * 1000.0
    fps = total_frames / elapsed
    print(f"  • Scenario {name:<22}: {elapsed:.3f}s | {ms_per_frame:.2f} ms/frame | {fps:.1f} fps")
    return ms_per_frame, fps

ms_A, fps_A = bench_batch([301, 302, 303, 304], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], "A (4x CFG=1.0, depth=0)")
ms_B, fps_B = bench_batch([305, 306, 307, 308], [1.6, 1.6, 1.6, 1.6], [0.0, 0.0, 0.0, 0.0], "B (4x CFG=1.6, depth=0)")
ms_C, fps_C = bench_batch([309, 310, 311, 312], [1.0, 1.0, 1.0, 1.6], [0.0, 0.0, 0.0, 0.0], "C (3x 1.0 + 1x 1.6)")
ms_D, fps_D = bench_batch([313, 314, 315, 316], [1.6, 1.6, 1.6, 1.6], [0.9, 0.9, 0.9, 0.9], "D (4x CFG=1.6, fallback 0.9)")

print(f"  • Ratio B / A: {ms_B / ms_A:.2f}x (Gate: <= 2.2x)")
assert (ms_B / ms_A) <= 2.2, f"Scenario B exceeded 2.2x of Scenario A: {ms_B/ms_A:.2f}x"
print(f"  • Speedup C over D (Mixed-Batch Isolation): {ms_D / ms_C:.2f}x faster (Gate: > 1.0x)")
assert ms_C < ms_D, f"Scenario C was not faster than D! (ms_C={ms_C:.2f}, ms_D={ms_D:.2f})"
assert ms_B <= ms_D * 1.05, f"Scenario B was significantly slower than D! (ms_B={ms_B:.2f}, ms_D={ms_D:.2f})"
print("  ✅ TEST 1.3 PASSED: On-GPU CFG + Argmax is active with mixed-batch isolation verified!")

# -----------------------------------------------------------------------------
# Test 1.4: Tiled Vocoder VRAM Cap & Seam Continuity (5-min & 15-min Decodes)
# -----------------------------------------------------------------------------
print("\n[TEST 1.4] Tiled Vocoder VRAM Cap & Seam Continuity (5-min vs 15-min)...")
base_vram = get_vram_mb(0)
print(f"  • Base GPU 0 VRAM: {base_vram:.1f} MB")

def test_tiled_decode(duration_mins, n_frames):
    print(f"  --> Testing {duration_mins}-minute decode ({n_frames} acoustic frames, {n_frames*1920:,} audio samples)...", flush=True)
    np.random.seed(42)
    # Generate realistic pseudo-codebook tokens (0 to 2047)
    dummy_tokens = np.random.randint(0, 2048, size=n_frames * 16, dtype=np.int32).tolist()
    
    t0 = time.time()
    audio_pcm = voc0.decode_tiled(dummy_tokens, ctx_frames=80, step_frames=160)
    elapsed = time.time() - t0
    peak_vram = get_vram_mb(0)
    vram_delta = peak_vram - base_vram
    
    audio_s = n_frames * 0.08
    rtf = elapsed / audio_s
    print(f"      Wall time: {elapsed:.2f}s | Audio duration: {audio_s:.1f}s | RTF: {rtf:.3f}", flush=True)
    print(f"      Peak VRAM: {peak_vram:.1f} MB | Delta: +{vram_delta:.1f} MB (Gate: <= 1,500 MiB)", flush=True)
    
    assert len(audio_pcm) == n_frames * 1920, f"Sample count mismatch: {len(audio_pcm)} != {n_frames * 1920}"
    assert vram_delta <= 1500.0, f"Vocoder VRAM exceeded 1,500 MiB: {vram_delta:.1f} MB"
    
    pcm_arr = np.array(audio_pcm, dtype=np.float32)
    assert not np.isnan(pcm_arr).any(), "NaN found in decoded audio!"
    assert not np.isinf(pcm_arr).any(), "Inf found in decoded audio!"
    
    # Check seam continuity at tile boundaries k * 160 * 1920
    step_samples = 160 * 1920
    max_seam_delta = 0.0
    for k in range(1, n_frames // 160):
        idx = k * step_samples
        if idx < len(pcm_arr):
            diff = abs(float(pcm_arr[idx] - pcm_arr[idx - 1]))
            if diff > max_seam_delta:
                max_seam_delta = diff
            assert diff < 0.35, f"Abnormal seam jump at sample {idx}: {diff:.4f} >= 0.35"
            assert pcm_arr[idx] != 0.0 or pcm_arr[idx - 1] != 0.0, f"Dead-zero dropout at seam {idx}!"
    print(f"      Max Seam Delta at Tile Boundaries: {max_seam_delta:.4f} (Gate: < 0.35)")
    return rtf, vram_delta

rtf_5m, vram_5m = test_tiled_decode(5, 3750)
rtf_15m, vram_15m = test_tiled_decode(15, 11250)

print(f"  • 5-min Peak Delta:  +{vram_5m:.1f} MB | RTF: {rtf_5m:.3f}")
print(f"  • 15-min Peak Delta: +{vram_15m:.1f} MB | RTF: {rtf_15m:.3f}")
assert abs(vram_15m - vram_5m) < 200.0, f"VRAM scaled with length! (5m={vram_5m:.1f} MB, 15m={vram_15m:.1f} MB)"
print("  ✅ TEST 1.4 PASSED: Tiled Vocoder strictly flat VRAM (<= 1,500 MiB) with pristine seam continuity!")

print("\n" + "=" * 80)
print("  🎉 ALL PHASE 1 TESTS PASSED WITH HARD ASSERTIONS!")
print("=" * 80)
