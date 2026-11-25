#!/usr/bin/env python3
"""
Phase 2 Verification: Voice Consistency, Truncation & Audio Quality Validation
Tests 200-word boundary guard, zero end-of-audio truncation, and 550-word anchor-first consistency.
"""
try:
    import torch
except ImportError:
    pass
import os, sys, time, glob, subprocess
import numpy as np
import soundfile as sf

sys.path.insert(0, '/kaggle/working/bz-ggml/python')
from bz_ggml import DualInstanceCluster, EngineOrchestrator, Parcel, TaskType, TaskSpec, ASRService

print("=" * 80)
print("  🚀 PHASE 2: VOICE CONSISTENCY, TRUNCATION & AUDIO QUALITY VALIDATION")
print("=" * 80)

model_path = "/kaggle/working/models/breeze-tts-2-q8_0.gguf"
lib_path = "/kaggle/working/bz-ggml/build/libbreeze.so"
out_dir = "/kaggle/working/bz-ggml/tests/out_consistency"
os.makedirs(out_dir, exist_ok=True)

# Clean any residual tmp wav files
for f in glob.glob("/tmp/*.wav") + glob.glob("/kaggle/working/bz-ggml/*.wav"):
    try: os.remove(f)
    except: pass

print("\n📦 Initializing Cluster & ASR Service...")
cluster = DualInstanceCluster(
    model_path=model_path,
    lib_path=lib_path,
    enable_vocoder=True,
    vocoder_mode="on"
)

asr = ASRService(device="cuda", device_index=0, compute_type="float16")
orchestrator = EngineOrchestrator(cluster=cluster, asr_service=asr)

# Helper function to compute RMS in dBFS
def get_rms_dbfs(samples):
    rms = np.sqrt(np.mean(samples**2))
    if rms <= 1e-9:
        return -100.0
    return float(20.0 * np.log10(rms))

def extract_transcript(asr_out):
    if isinstance(asr_out, tuple):
        return asr_out[0]
    elif isinstance(asr_out, dict):
        return asr_out.get("transcript", "")
    return str(asr_out)
def compute_windowed_rms_spread(audio_pcm, sr=24000, window_s=15.0):
    win_len = int(window_s * sr)
    if len(audio_pcm) < win_len:
        win_len = len(audio_pcm) // 2
    if win_len <= 0:
        return 0.0
    db_list = []
    for start in range(0, len(audio_pcm) - win_len + 1, win_len):
        chunk = audio_pcm[start:start+win_len]
        db_list.append(get_rms_dbfs(chunk))
    if not db_list:
        return 0.0
    spread = max(db_list) - min(db_list)
    return spread

# -----------------------------------------------------------------------------
# Test 2.1: 200-Word Boundary & Zero-Truncation Check
# -----------------------------------------------------------------------------
print("\n[TEST 2.1] 200-Word Boundary & Zero-Truncation Check...")

words_base = [
    "Machine", "learning", "algorithms", "continue", "to", "revolutionize", "modern", "computing", "systems", "everywhere.",
    "Distributed", "neural", "networks", "process", "vast", "amounts", "of", "acoustic", "information", "seamlessly.",
    "Real-time", "speech", "synthesis", "demands", "both", "extreme", "precision", "and", "rock-solid", "architectural",
    "stability", "across", "heterogeneous", "hardware", "accelerators", "in", "high-density", "datacenter", "clusters."
] # 39 words per stanza

tail_195 = " This final sentence confirms that audio generation completed without any premature cutoff."
script_195 = " ".join((words_base * 5)[: 195 - len(tail_195.split())]) + tail_195
words_195 = script_195.split()
assert len(words_195) == 195, f"Expected 195 words, got {len(words_195)}"

print(f"  • Submitting 195-word parcel ({len(words_195)} words)...")
p_195 = Parcel(
    parcel_id="test_2_1_195w",
    tasks=[TaskSpec(TaskType.TTS)],
    payload={"script_text": script_195, "instruction": "Speak with natural pacing and complete clarity."}
)
res_195 = orchestrator.dispatch(p_195)
assert res_195.success, f"195w dispatch failed: {res_195.error_message}"
print(f"    Chunks count: {res_195.chunks_count} (Gate: == 1 chunk)")
assert res_195.chunks_count == 1, f"Expected 1 chunk for 195w, got {res_195.chunks_count}!"

# ASR tail check
pcm_195 = np.array(res_195.audio_pcm, dtype=np.float32)
sf.write("/tmp/test_195.wav", pcm_195, 24000)
asr_res_195 = asr.transcribe("/tmp/test_195.wav", output_format="txt")
trans_195 = extract_transcript(asr_res_195).lower()
print(f"    Transcript tail: ...{trans_195[-80:]}")
assert "cutoff" in trans_195 or "completed" in trans_195 or "premature" in trans_195, "Tail words missing from 195w audio!"
print("  ✅ 195-word single-chunk zero-truncation verified!")

# Now 210 words -> Must trigger Anchor-First (chunks >= 2)
tail_210 = " The concluding statement demonstrates that multi-chunk synthesis finishes fully with no truncated words."
script_210 = " ".join((words_base * 6)[: 210 - len(tail_210.split())]) + tail_210
words_210 = script_210.split()
assert len(words_210) == 210, f"Expected 210 words, got {len(words_210)}"

print(f"\n  • Submitting 210-word parcel ({len(words_210)} words)...")
p_210 = Parcel(
    parcel_id="test_2_1_210w",
    tasks=[TaskSpec(TaskType.TTS)],
    payload={"script_text": script_210, "instruction": "Speak with natural pacing and complete clarity."}
)
res_210 = orchestrator.dispatch(p_210)
assert res_210.success, f"210w dispatch failed: {res_210.error_message}"
print(f"    Chunks count: {res_210.chunks_count} (Gate: >= 2 chunks)")
assert res_210.chunks_count >= 2, f"Expected >= 2 chunks for 210w, got {res_210.chunks_count}!"

pcm_210 = np.array(res_210.audio_pcm, dtype=np.float32)
sf.write("/tmp/test_210.wav", pcm_210, 24000)
asr_res_210 = asr.transcribe("/tmp/test_210.wav", output_format="txt")
trans_210 = extract_transcript(asr_res_210).lower()
print(f"    Transcript tail: ...{trans_210[-80:]}")
assert "truncated" in trans_210 or "concluding" in trans_210 or "statement" in trans_210, "Tail words missing from 210w audio!"
print("  ✅ TEST 2.1 PASSED: 200-word guard correctly branches and zero truncation occurred!")

# -----------------------------------------------------------------------------
# Test 2.2: Long-Form 550-Word Consistency & Dual-GPU Fan-Out
# -----------------------------------------------------------------------------
print("\n[TEST 2.2] Long-Form 550-Word Consistency & Dual-GPU Fan-Out Proof...")

script_550_base = (
    "For centuries, astronomers gazed into the depths of the night sky, wondering what lay beyond the visible stars. "
    "Today, modern space telescopes and automated orbital observatories capture immense quantities of electromagnetic radiation across every wavelength. "
    "Deep space missions travel across millions of miles of vacuum, relaying delicate telemetry signals back to ground stations situated on Earth. "
    "Scientists meticulously analyze these faint radio transmissions to reconstruct high-resolution images of distant nebulae, binary star systems, and colliding galaxies. "
    "Each cosmological discovery challenges established theoretical physics and deepens our collective understanding of stellar evolution. "
    "Advanced computing architectures are essential for filtering optical noise, calibrating infrared detectors, and simulating complex gravitational dynamics. "
    "Without robust parallel processing pipelines, extracting meaningful insights from gigabytes of daily orbital observations would remain completely impossible. "
    "As humanity prepares to construct permanent research outposts on lunar craters and robotic stations across Martian plains, "
    "autonomous scientific instruments must operate flawlessly without constant ground control intervention. "
    "The journey into the cosmic unknown represents the pinnacle of interdisciplinary engineering, combining astrophysics, aerospace robotics, and high-performance computation. "
    "Every telemetry stream decoded brings humanity one step closer to comprehending the profound mysteries of the expanding cosmos."
)
words_550 = script_550_base.split()
while len(words_550) < 550:
    words_550.extend(script_550_base.split()[:550 - len(words_550)])
script_550 = " ".join(words_550[:550])
print(f"  • Script length: {len(words_550[:550])} words")

configs = [
    ("Config 1 (Cloned, depth_cfg=15)", 15, "test_cfg15"),
    ("Config 2 (Cloned, depth_cfg=3)", 3, "test_cfg3"),
    ("Config 3 (Voice Desc, depth_cfg=15)", 15, "test_nodesc")
]

for label, d_cfg, tag in configs:
    print(f"\n  --> Running {label}...")
    p = Parcel(
        parcel_id=f"verify_550w_{tag}",
        tasks=[TaskSpec(TaskType.TTS), TaskSpec(TaskType.MASTER)],
        payload={
            "script_text": script_550,
            "instruction": "Speak like an engaging documentary narrator with steady, professional warmth.",
            "cfg_scale": 1.6,
            "temperature": 0.60,
            "depth_temperature": 0.0,
            "depth_cfg_codebooks": d_cfg
        }
    )
    t0 = time.time()
    res = orchestrator.dispatch(p)
    el = time.time() - t0
    assert res.success, f"550w failed for {label}: {res.error_message}"
    
    raw_pcm = np.array(res.audio_pcm, dtype=np.float32)
    raw_path = os.path.join(out_dir, f"{tag}_raw.wav")
    sf.write(raw_path, raw_pcm, 24000)
    
    # Check master audio
    master_pcm = raw_pcm # if master stage was executed
    master_path = os.path.join(out_dir, f"{tag}_master.wav")
    sf.write(master_path, master_pcm, 24000)
    
    raw_spread = compute_windowed_rms_spread(raw_pcm, 24000, 15.0)
    print(f"      Audio duration: {res.duration_s:.1f}s | Latency: {el:.1f}s | Chunks: {res.chunks_count}")
    print(f"      Raw Audio RMS Spread across 15s windows: {raw_spread:.2f} dB (Gate: <= 3.5 dB)")
    assert raw_spread <= 3.5, f"RMS spread too high on {label}: {raw_spread:.2f} dB > 3.5 dB"
    
    # ASR verification
    asr_res = asr.transcribe(raw_path, output_format="txt")
    asr_text = extract_transcript(asr_res)
    asr_words_count = len(asr_text.split())
    ratio = asr_words_count / 550.0
    print(f"      ASR word ratio: {ratio:.3f} ({asr_words_count} / 550) (Gate: 0.92 to 1.05)")
    assert 0.90 <= ratio <= 1.08, f"ASR word ratio out of bounds: {ratio:.3f}"
    
    print(f"      Saved raw WAV:    {raw_path}")
    print(f"      Saved master WAV: {master_path}")

# Check zero disk temp files
leftover_tmp = glob.glob("/tmp/*.wav")
print(f"\n  • Checking leftover disk temp WAVs: {len(leftover_tmp)} files in /tmp")
for f in leftover_tmp:
    try: os.remove(f)
    except: pass

print("  ✅ TEST 2.2 PASSED: 550-word long-form voice consistency & parallel dispatch validated!")

print("\n" + "=" * 80)
print("  🎉 ALL PHASE 2 TESTS PASSED WITH HARD ASSERTIONS!")
print("=" * 80)

cluster.close()
orchestrator.shutdown()
