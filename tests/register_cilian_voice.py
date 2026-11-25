"""
Phase 1 Subprocess: Official Voice Persona Registration for cilian_9sec.mp3
Uses EngineOrchestrator + ASRService (Whisper Turbo on GPU 0) + Resemble Enhancer (GPU 1)
+ FFmpeg Studio Mastering (gentle_loudnorm) + MimiCodec Audio Encoder.
Persists cilian.breeze and prints verification diagnostics, then exits so CUDA reclaims 100% VRAM.
"""

import os
import sys
import json
import time

sys.path.insert(0, "/kaggle/working/bz-ggml/python")
os.chdir("/kaggle/working/bz-ggml")

from bz_ggml import (
    ASRService,
    DualInstanceCluster,
    EngineOrchestrator,
    Parcel,
    deserialize_breeze_voice,
)

MODEL_PATH = "/kaggle/working/models/breeze-tts-2-f16.gguf"
LIB_PATH = "/kaggle/working/bz-ggml/build/libbreeze.so"
AUDIO_INPUT = "/kaggle/working/cilian_9sec.mp3"
RESULTS_DIR = "/kaggle/working/benchmark_results"
OUT_BREEZE = os.path.join(RESULTS_DIR, "cilian.breeze")
OUT_META = os.path.join(RESULTS_DIR, "cilian_registration_meta.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 80, flush=True)
    print(">>> PHASE 1: OFFICIAL VOICE REGISTRATION (cilian_9sec.mp3)", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    asr_service = ASRService(
        model_size="turbo",
        device="cuda",
        device_index=0,
        compute_type="float16",
    )
    cluster = DualInstanceCluster(
        model_path=MODEL_PATH,
        lib_path=LIB_PATH,
        enable_q4_burst=False,
        enable_asr=True,
        enable_enhancer=True,
        precision="fp16",
    )
    orchestrator = EngineOrchestrator(cluster=cluster, asr_service=asr_service)

    try:
        parcel = Parcel.voice_registration(
            voice_name="cilian",
            audio_input=AUDIO_INPUT,
            transcript=None,
            enhance="auto",
            master=True,
            max_duration_s=10.0,
            parcel_id="reg_cilian_9sec",
        )
        print(f">>> Submitting registration pipeline tasks: {[t.task_type.value for t in parcel.tasks]}", flush=True)
        res = orchestrator.dispatch(parcel, timeout=300.0)
        if not res.success or res.voice_bytes is None:
            raise RuntimeError(f"Voice registration failed: {res.error_message}")

        with open(OUT_BREEZE, "wb") as f:
            f.write(res.voice_bytes)

        meta = deserialize_breeze_voice(res.voice_bytes)
        reg_info = {
            "voice_name": "cilian",
            "breeze_path": OUT_BREEZE,
            "file_bytes": len(res.voice_bytes),
            "frames": meta["frames"],
            "audio_seconds": round(meta["frames"] * 0.08, 2),
            "code_count": len(meta["codes"]),
            "sample_rate": meta["sample_rate"],
            "transcript": meta["text"],
            "stage_timings": res.stage_timings,
            "total_wall_s": round(time.time() - t0, 2),
        }
        with open(OUT_META, "w") as f:
            json.dump(reg_info, f, indent=2)

        print("\n>>> VOICE REGISTRATION SUCCEEDED:", flush=True)
        print(f"  Saved Container : {OUT_BREEZE} ({len(res.voice_bytes)/1024.0:.1f} KB)", flush=True)
        print(f"  Transcript      : \"{meta['text']}\"", flush=True)
        print(f"  Acoustic Frames : {meta['frames']} ({meta['frames']*0.08:.2f}s @ 12.5 Hz)", flush=True)
        print(f"  Discrete Tokens : {len(meta['codes'])} tokens (16 codebooks)", flush=True)
        print(f"  Stage Timings   : {res.stage_timings}", flush=True)
    finally:
        try:
            orchestrator.shutdown()
        except Exception:
            pass
        try:
            cluster.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
