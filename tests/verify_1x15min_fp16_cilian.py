"""
Phase 2: 1x 15-Minute Full Cloned FP16 TTS Quality Benchmark (Cilian Voice, enhance=False on output).
Evaluates maximum raw FP16 cloned quality with tuned consistency, diversity, and depth warmth:
  - Uses pre-registered cilian.breeze (from Parcel.voice_registration).
  - Model: breeze-tts-2-f16.gguf (FP16 precision, 19,500 tokens/GPU symmetric).
  - Parameters: cfg_scale=2.2, follower_cfg_scale=2.2, temperature=0.80, top_p=0.92, top_k=50,
                repetition_penalty=1.1, depth_temperature=0.75, depth_cfg_codebooks=15.
  - Output: enhance=False (raw codec quality preserved), master=True (also saves untouched raw PCM!).
  - Continuous 500 ms telemetry, pitch F0 & loudness diagnostics, FLAC + WAV exports.
"""

import os
import sys
import time
import json
import wave
import threading
import subprocess
from typing import List, Dict, Any, Optional

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/kaggle/working/bz-ggml/python")
sys.path.insert(0, "/kaggle/working/bz-ggml/tests")
os.chdir("/kaggle/working/bz-ggml")

from bz_ggml import (
    DualInstanceCluster,
    EngineOrchestrator,
    Parcel,
    ParcelResult,
    deserialize_breeze_voice,
)
from texts_corpus_5x20m import get_essay_script

MODEL_PATH = "/kaggle/working/models/breeze-tts-2-f16.gguf"
LIB_PATH = "/kaggle/working/bz-ggml/build/libbreeze.so"
RESULTS_DIR = "/kaggle/working/benchmark_results"
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio")
BREEZE_PATH = os.path.join(RESULTS_DIR, "cilian.breeze")
LOG_FILE_PATH = os.path.join(RESULTS_DIR, "benchmark_1x15m_fp16_cilian_live.log")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

log_file_handle = open(LOG_FILE_PATH, "w", encoding="utf-8")


def log_print(msg: str = "", flush: bool = True):
    print(msg, flush=flush)
    try:
        log_file_handle.write(msg + "\n")
        if flush:
            log_file_handle.flush()
    except Exception:
        pass


def analyze_audio_quality(pcm: np.ndarray, sr: int = 24000) -> Dict[str, float]:
    if pcm is None or len(pcm) == 0:
        return {}
    arr = pcm.astype(np.float32)
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    rms_db = float(20.0 * np.log10(rms + 1e-9))

    # Measure pitch F0 on first 45s voiced windows via autocorrelation
    seg = arr[: min(len(arr), 45 * sr)]
    hop, win = 480, 1920
    f0s = []
    sil_thresh = max(0.005, rms * 0.4)
    for i in range(0, len(seg) - win, hop):
        frame = seg[i : i + win]
        if np.sqrt(np.mean(frame ** 2)) < sil_thresh:
            continue
        frame = frame - np.mean(frame)
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        min_lag, max_lag = sr // 300, sr // 65
        if max_lag < len(ac) and ac[0] > 0:
            lag = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
            if ac[lag] > 0.35 * ac[0]:
                f0s.append(sr / lag)

    return {
        "duration_s": round(len(arr) / float(sr), 2),
        "peak_amp": round(peak, 4),
        "rms_dbfs": round(rms_db, 2),
        "voiced_frames_sampled": len(f0s),
        "mean_f0_hz": round(float(np.mean(f0s)), 1) if f0s else 0.0,
        "std_f0_hz": round(float(np.std(f0s)), 1) if f0s else 0.0,
    }


def analyze_chunk_drift(pcm: np.ndarray, sr: int = 24000, pause_ms: int = 200) -> Dict[str, Any]:
    """Splits stitched audio at exact zero breath gaps and measures per-chunk F0/RMS/centroid stability."""
    if pcm is None or len(pcm) == 0:
        return {}
    arr = pcm.astype(np.float32)
    min_zero_len = max(1, int(sr * ((pause_ms - 10) / 1000.0)))
    bounds: List[tuple] = []
    i, n = 0, len(arr)
    cur_start = 0
    while i < n - min_zero_len:
        if np.all(np.abs(arr[i : i + min_zero_len]) < 1e-6):
            j = i + min_zero_len
            while j < n and abs(arr[j]) < 1e-6:
                j += 1
            if i > cur_start:
                bounds.append((cur_start, i))
            cur_start = j
            i = j
        else:
            i += 480
    if cur_start < n:
        bounds.append((cur_start, n))

    hop, win = int(0.02 * sr), int(0.08 * sr)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    hann = np.hanning(win)
    chunk_f0s, chunk_rmss, chunk_cents, chunk_harms = [], [], [], []

    for s, e in bounds:
        seg = arr[s:e]
        if len(seg) < win * 2:
            continue
        rms_db = float(20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-9))
        f0s, cents, harms = [], [], []
        for k in range(0, len(seg) - win, hop):
            fr = seg[k : k + win]
            if float(np.sqrt(np.mean(fr ** 2))) < 0.008:
                continue
            mag = np.abs(np.fft.rfft(fr * hann))
            ms = float(np.sum(mag) + 1e-9)
            cents.append(float(np.sum(freqs * mag) / ms))
            fr0 = fr - np.mean(fr)
            ac = np.correlate(fr0, fr0, mode="full")[win - 1 :]
            if ac[0] > 0:
                lag = int(np.argmax(ac[sr // 300 : sr // 65]) + sr // 300)
                ratio = float(ac[lag] / ac[0])
                harms.append(ratio)
                if ratio > 0.35:
                    f0s.append(sr / lag)
        if f0s:
            chunk_f0s.append(float(np.mean(f0s)))
        chunk_rmss.append(rms_db)
        if cents:
            chunk_cents.append(float(np.mean(cents)))
        if harms:
            chunk_harms.append(float(np.mean(harms)))

    return {
        "stitched_segments_detected": len(bounds),
        "f0_range_hz": round(max(chunk_f0s) - min(chunk_f0s), 2) if chunk_f0s else 0.0,
        "f0_std_across_chunks_hz": round(float(np.std(chunk_f0s)), 2) if chunk_f0s else 0.0,
        "rms_range_db": round(max(chunk_rmss) - min(chunk_rmss), 2) if chunk_rmss else 0.0,
        "rms_std_across_chunks_db": round(float(np.std(chunk_rmss)), 2) if chunk_rmss else 0.0,
        "centroid_range_hz": round(max(chunk_cents) - min(chunk_cents), 1) if chunk_cents else 0.0,
        "centroid_std_across_chunks_hz": round(float(np.std(chunk_cents)), 1) if chunk_cents else 0.0,
        "mean_harmonicity": round(float(np.mean(chunk_harms)), 3) if chunk_harms else 0.0,
    }


class SystemTelemetryLogger:
    def __init__(self, orchestrator: EngineOrchestrator, output_csv: str, interval_s: float = 0.5):
        self.orchestrator = orchestrator
        self.output_csv = output_csv
        self.interval_s = interval_s
        self.records: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._csv_file = open(self.output_csv, "w", encoding="utf-8")
        self._csv_headers_written = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

    def _get_nvidia_smi(self) -> Dict[str, float]:
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            stats = {}
            for line in res.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 5:
                    idx = int(parts[0])
                    stats[f"gpu{idx}_util"] = float(parts[1])
                    stats[f"gpu{idx}_used_mb"] = float(parts[2])
                    stats[f"gpu{idx}_total_mb"] = float(parts[3])
                    stats[f"gpu{idx}_temp"] = float(parts[4])
            return stats
        except Exception:
            return {}

    def _run(self):
        t0 = time.time()
        while not self._stop_event.is_set():
            t_now = time.time() - t0
            smi = self._get_nvidia_smi()
            tok_gpu0, tok_gpu1, ceil_gpu0, ceil_gpu1 = 0, 0, 19500, 19500
            try:
                coord = getattr(self.orchestrator.cluster, "dynamic_coord", None)
                if coord:
                    with coord.lock:
                        tok_gpu0 = coord.active_tokens.get(0, 0)
                        tok_gpu1 = coord.active_tokens.get(1, 0)
                        ceil_gpu0 = getattr(coord, "gpu0_ceiling", getattr(coord, "ceiling_gpu0", 19500))
                        ceil_gpu1 = getattr(coord, "gpu1_ceiling", getattr(coord, "ceiling_gpu1_normal", 19500))
            except Exception:
                pass
            pending_chunks = getattr(self.orchestrator, "_pending_tts_chunks", 0)
            rec = {
                "elapsed_s": round(t_now, 2),
                "gpu0_util": smi.get("gpu0_util", 0.0),
                "gpu1_util": smi.get("gpu1_util", 0.0),
                "gpu0_used_mb": smi.get("gpu0_used_mb", 0.0),
                "gpu1_used_mb": smi.get("gpu1_used_mb", 0.0),
                "gpu0_free_mb": smi.get("gpu0_total_mb", 15360.0) - smi.get("gpu0_used_mb", 0.0),
                "gpu1_free_mb": smi.get("gpu1_total_mb", 15360.0) - smi.get("gpu1_used_mb", 0.0),
                "tok_gpu0": tok_gpu0,
                "tok_gpu1": tok_gpu1,
                "ceil_gpu0": ceil_gpu0,
                "ceil_gpu1": ceil_gpu1,
                "pending_chunks": pending_chunks,
            }
            self.records.append(rec)
            try:
                if not self._csv_headers_written:
                    self._csv_file.write(",".join(rec.keys()) + "\n")
                    self._csv_headers_written = True
                self._csv_file.write(",".join(str(rec[k]) for k in rec.keys()) + "\n")
                self._csv_file.flush()
            except Exception:
                pass
            time.sleep(self.interval_s)


def run_benchmark():
    log_print("=" * 80)
    log_print(">>> PHASE 2: FULL 15-MIN FP16 CLONED CILIAN QUALITY BENCHMARK...")
    log_print("=" * 80)

    if not os.path.exists(BREEZE_PATH):
        raise FileNotFoundError(f"Missing registered voice file: {BREEZE_PATH}")

    with open(BREEZE_PATH, "rb") as f:
        v_meta = deserialize_breeze_voice(f.read())

    log_print(f">>> Loaded Registered Voice: {BREEZE_PATH}")
    log_print(f"    Transcript : \"{v_meta['text']}\"")
    log_print(f"    Ref Frames : {v_meta['frames']} ({v_meta['frames']*0.08:.2f}s @ {v_meta['sample_rate']} Hz)")

    cluster = DualInstanceCluster(
        model_path=MODEL_PATH,
        lib_path=LIB_PATH,
        enable_q4_burst=False,
        enable_asr=False,
        enable_enhancer=False,
        precision="fp16",
    )
    orchestrator = EngineOrchestrator(cluster=cluster)

    telemetry_csv = os.path.join(RESULTS_DIR, "telemetry_1x15m_fp16_cilian.csv")
    telemetry = SystemTelemetryLogger(orchestrator, telemetry_csv, interval_s=0.5)

    req_id = "req_15m_cilian_fp16"
    target_words = 2100
    text = get_essay_script(0, target_words=target_words)
    actual_words = len(text.split())

    instruction = "Speak clearly and naturally."

    parcel = Parcel.tts_pipeline(
        script_text=text,
        voice_name="cilian",
        instruction=instruction,
        cfg_scale=1.6,
        follower_cfg_scale=1.6,
        temperature=0.55,
        top_p=0.85,
        top_k=30,
        repetition_penalty=1.08,
        depth_temperature=0.0,
        depth_cfg_codebooks=3,
        seed=42,
        pause_ms=200,
        enhance=False,
        master=True,
        srt=False,
        ref_codes=v_meta["codes"],
        ref_frames=v_meta["frames"],
        ref_text=v_meta["text"],
        parcel_id=req_id,
    )

    log_print(f"\n>>> CONFIGURED LOCKED-CONSISTENCY CLONING PARAMETERS (Option A):")
    log_print(f"  ID                  : {req_id}")
    log_print(f"  Words               : {actual_words} (~45w/chunk coherent horizon)")
    log_print(f"  Model Precision     : FP16 (19,500 tokens/GPU)")
    log_print(f"  CFG / Follower CFG  : 1.6 / 1.6 (Neutral instruction)")
    log_print(f"  Backbone Sampling   : temp=0.55, top_p=0.85, top_k=30, rep_pen=1.08")
    log_print(f"  Depth Decoder       : depth_temperature=0.0 (on-GPU argmax), depth_cfg=3 (Timbre-Lock)")
    log_print(f"  Vocoding & Stitch   : Isolated per-chunk vocode + +-2.5dB RMS leveling + 200ms pause")
    log_print(f"  Output Processing   : enhance=False (pure FP16 raw + mastered -16 LUFS)")
    log_print("=" * 80)

    telemetry.start()
    t_start = time.time()
    completed_result: Optional[ParcelResult] = None
    wall_s = 0.0

    def _dispatch_worker():
        nonlocal completed_result, wall_s
        t0 = time.time()
        completed_result = orchestrator.dispatch(parcel, timeout=3600.0)
        wall_s = time.time() - t0

    try:
        t_worker = threading.Thread(target=_dispatch_worker, daemon=True)
        t_worker.start()
        while t_worker.is_alive():
            t_worker.join(timeout=5.0)
            t_el = time.time() - t_start
            if telemetry.records:
                lt = telemetry.records[-1]
                log_print(
                    f"  [t={t_el:6.1f}s] Active Tokens: GPU0={lt['tok_gpu0']:,}/{lt['ceil_gpu0']:,}, "
                    f"GPU1={lt['tok_gpu1']:,}/{lt['ceil_gpu1']:,} | "
                    f"Util: GPU0={lt['gpu0_util']:3.0f}%, GPU1={lt['gpu1_util']:3.0f}% | "
                    f"Free VRAM: GPU0={lt['gpu0_free_mb']:.0f}MB, GPU1={lt['gpu1_free_mb']:.0f}MB | "
                    f"Pending: {lt['pending_chunks']}"
                )
    finally:
        telemetry.stop()
        try:
            orchestrator.shutdown()
        except Exception:
            pass
        try:
            cluster.close()
        except Exception:
            pass

    if completed_result is None or not completed_result.success:
        err = completed_result.error_message if completed_result else "Execution aborted"
        log_print(f"\n[-] BENCHMARK FAILED: {err}")
        sys.exit(1)

    sr = getattr(completed_result, "sample_rate", 24000)
    raw_pcm = getattr(completed_result, "raw_audio_pcm", None)
    master_pcm = getattr(completed_result, "audio_pcm", None)
    if raw_pcm is None:
        raw_pcm = master_pcm

    raw_wav = os.path.join(AUDIO_DIR, f"{req_id}_raw.wav")
    master_wav = os.path.join(AUDIO_DIR, f"{req_id}_mastered.wav")
    raw_flac = os.path.join(AUDIO_DIR, f"{req_id}_raw.flac")
    master_flac = os.path.join(AUDIO_DIR, f"{req_id}_mastered.flac")

    sf.write(raw_wav, raw_pcm, 24000, subtype="PCM_16")
    sf.write(raw_flac, raw_pcm, 24000, format="FLAC")
    sf.write(master_wav, master_pcm, sr, subtype="PCM_16")
    sf.write(master_flac, master_pcm, sr, format="FLAC")

    audio_dur = len(master_pcm) / float(sr)
    rtf_speedup = audio_dur / max(0.001, wall_s)
    rtf_fraction = wall_s / max(0.001, audio_dur)

    raw_diag = analyze_audio_quality(raw_pcm, 24000)
    raw_drift = analyze_chunk_drift(raw_pcm, 24000, pause_ms=200)
    master_diag = analyze_audio_quality(master_pcm, sr)

    recs = telemetry.records
    u0 = [r["gpu0_util"] for r in recs] if recs else [0.0]
    u1 = [r["gpu1_util"] for r in recs] if recs else [0.0]
    f0 = [r["gpu0_free_mb"] for r in recs] if recs else [0.0]
    f1 = [r["gpu1_free_mb"] for r in recs] if recs else [0.0]

    summary = {
        "benchmark": "1x 15-Minute Full Cloned Cilian FP16 Quality & Consistency Benchmark (Option A)",
        "request_id": req_id,
        "voice_name": "cilian",
        "voice_transcript": v_meta["text"],
        "voice_ref_frames": v_meta["frames"],
        "words": actual_words,
        "chunks_count": completed_result.chunks_count,
        "model": "breeze-tts-2-f16.gguf",
        "precision": "fp16",
        "cfg_scale": 1.6,
        "follower_cfg_scale": 1.6,
        "temperature": 0.55,
        "top_p": 0.85,
        "top_k": 30,
        "repetition_penalty": 1.08,
        "depth_temperature": 0.0,
        "depth_cfg_codebooks": 3,
        "output_enhance": False,
        "output_master": True,
        "audio_seconds": round(audio_dur, 2),
        "audio_minutes": round(audio_dur / 60.0, 2),
        "wall_seconds": round(wall_s, 2),
        "wall_minutes": round(wall_s / 60.0, 2),
        "speedup_xrealtime": round(rtf_speedup, 2),
        "rtf_fraction": round(rtf_fraction, 4),
        "raw_audio_metrics": raw_diag,
        "raw_cross_chunk_drift": raw_drift,
        "mastered_audio_metrics": master_diag,
        "gpu0_avg_compute_util_pct": round(float(np.mean(u0)), 2),
        "gpu1_avg_compute_util_pct": round(float(np.mean(u1)), 2),
        "gpu0_p50_util_pct": round(float(np.median(u0)), 2),
        "gpu1_p50_util_pct": round(float(np.median(u1)), 2),
        "gpu0_p90_util_pct": round(float(np.percentile(u0, 90)), 2),
        "gpu1_p90_util_pct": round(float(np.percentile(u1, 90)), 2),
        "gpu0_p99_util_pct": round(float(np.percentile(u0, 99)), 2),
        "gpu1_p99_util_pct": round(float(np.percentile(u1, 99)), 2),
        "min_free_vram_gpu0_mib": round(float(np.min(f0)), 1),
        "min_free_vram_gpu1_mib": round(float(np.min(f1)), 1),
        "stage_timings": completed_result.stage_timings,
    }

    with open(os.path.join(RESULTS_DIR, "summary_1x15m_fp16_cilian.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log_print("\n" + "=" * 80)
    log_print(">>> FULL 15-MIN CLONED CILIAN BENCHMARK COMPLETED!")
    log_print("=" * 80)
    log_print(f"  Chunks Generated : {completed_result.chunks_count}")
    log_print(f"  Audio Duration   : {audio_dur:.2f}s ({audio_dur/60.0:.2f} min)")
    log_print(f"  Wall Time        : {wall_s:.2f}s ({wall_s/60.0:.2f} min)")
    log_print(f"  Speedup          : {rtf_speedup:.2f}x realtime (RTF {rtf_fraction:.4f})")
    log_print(f"  Raw Audio Stats  : {raw_diag}")
    log_print(f"  Raw Chunk Drift  : {raw_drift}")
    log_print(f"  Mastered Stats   : {master_diag}")
    log_print(f"  GPU 0 Compute    : Mean={summary['gpu0_avg_compute_util_pct']:.2f}%, P50={summary['gpu0_p50_util_pct']:.1f}%, P90={summary['gpu0_p90_util_pct']:.1f}%, P99={summary['gpu0_p99_util_pct']:.1f}%")
    log_print(f"  GPU 1 Compute    : Mean={summary['gpu1_avg_compute_util_pct']:.2f}%, P50={summary['gpu1_p50_util_pct']:.1f}%, P90={summary['gpu1_p90_util_pct']:.1f}%, P99={summary['gpu1_p99_util_pct']:.1f}%")
    log_print(f"  VRAM Min Free    : GPU0={summary['min_free_vram_gpu0_mib']:.1f} MiB, GPU1={summary['min_free_vram_gpu1_mib']:.1f} MiB")
    log_print("=" * 80)

    try:
        t_arr = np.array([r["elapsed_s"] for r in recs])
        u0_arr, u1_arr = np.array(u0), np.array(u1)
        f0_arr, f1_arr = np.array(f0), np.array(f1)
        tok0_arr = np.array([r["tok_gpu0"] for r in recs])
        tok1_arr = np.array([r["tok_gpu1"] for r in recs])

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(t_arr, u0_arr, label="GPU 0 Util %", alpha=0.85)
        axs[0, 0].plot(t_arr, u1_arr, label="GPU 1 Util %", alpha=0.85)
        axs[0, 0].set_title("GPU Compute Utilization (Option A Cloned FP16, depth=0.0, dcfg=3)", fontweight="bold")
        axs[0, 0].set_ylim(0, 105)
        axs[0, 0].legend(loc="lower right")

        axs[0, 1].plot(t_arr, f0_arr, label="GPU 0 Free VRAM (MiB)")
        axs[0, 1].plot(t_arr, f1_arr, label="GPU 1 Free VRAM (MiB)")
        axs[0, 1].axhline(500, color="red", linestyle=":", label="500 MiB Margin")
        axs[0, 1].set_title("VRAM Free Headroom Over Time", fontweight="bold")
        axs[0, 1].legend(loc="lower left")

        axs[1, 0].plot(t_arr, tok0_arr, label="GPU 0 Active Tokens")
        axs[1, 0].plot(t_arr, tok1_arr, label="GPU 1 Active Tokens")
        axs[1, 0].axhline(19500, color="black", linestyle="--", label="19,500 FP16 Ceiling")
        axs[1, 0].set_title("Active Token Timeline", fontweight="bold")
        axs[1, 0].legend(loc="upper right")

        metrics = ["Mean", "P50", "P90", "P95", "P99"]
        vals_g0 = [np.mean(u0_arr), np.median(u0_arr), np.percentile(u0_arr, 90), np.percentile(u0_arr, 95), np.percentile(u0_arr, 99)]
        vals_g1 = [np.mean(u1_arr), np.median(u1_arr), np.percentile(u1_arr, 90), np.percentile(u1_arr, 95), np.percentile(u1_arr, 99)]
        x = np.arange(len(metrics))
        w = 0.35
        axs[1, 1].bar(x - w / 2, vals_g0, width=w, label="GPU 0")
        axs[1, 1].bar(x + w / 2, vals_g1, width=w, label="GPU 1")
        axs[1, 1].set_title("Compute Percentiles", fontweight="bold")
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(metrics)
        axs[1, 1].set_ylim(0, 105)
        axs[1, 1].legend(loc="lower right")

        plt.tight_layout()
        plot_out = os.path.join(RESULTS_DIR, "benchmark_1x15m_fp16_cilian_dashboard.png")
        fig.savefig(plot_out, dpi=180)
        plt.close(fig)
        log_print(f">>> Generated dashboard: {plot_out}")
    except Exception as pe:
        log_print(f"[-] Plotting exception: {pe}")


if __name__ == "__main__":
    run_benchmark()
