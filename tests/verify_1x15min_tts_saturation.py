"""
1x 15-Minute Isolated TTS Benchmark on Dual Tesla T4 GPUs.
Evaluates single-branch (CFG = 1.0) streaming performance:
  - 1 request of ~2,000 words (~15 minutes audio).
  - Authentic human English text (Astronomy & Cosmic Horizon).
  - cfg_scale = 1.0, follower_cfg_scale = 1.0 (Zero unconditional branch evaluation).
  - enable_asr = False (Dynamic token ceiling = 33,100 on both GPU 0 and GPU 1).
  - Pipelined vocoder: Step 480 (Context 80) dual-GPU streaming.
  - Audio saved as 24kHz 16-bit WAV to /kaggle/working/benchmark_results/audio/req_1x15m_cfg1.wav.
  - High-frequency continuous telemetry (500ms): GPU compute util %, VRAM headroom, active tokens.
"""

import sys, os, time, math, threading, json, subprocess, wave
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib
importlib.invalidate_caches()
sys.path_importer_cache.clear()
sys.path.insert(0, "/kaggle/working/bz-ggml/python")
sys.path.insert(0, "/kaggle/working/bz-ggml/tests")
os.chdir("/kaggle/working/bz-ggml")

from bz_ggml import (
    DualInstanceCluster,
    EngineOrchestrator,
    TaskType, TaskSpec, Parcel, ParcelResult,
)
from texts_corpus_5x20m import get_essay_script

MODEL_PATH = "/kaggle/working/models/breeze-tts-2-q8_0.gguf"
LIB_PATH = "/kaggle/working/bz-ggml/build/libbreeze.so"
RESULTS_DIR = "/kaggle/working/benchmark_results"
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio")
LOG_FILE_PATH = os.path.join(RESULTS_DIR, "benchmark_1x15m_live.log")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------------------
# DUAL TEE LOGGER (CONSOLE + PERSISTENT DISK LOG)
# -------------------------------------------------------------
log_file_handle = open(LOG_FILE_PATH, "a", encoding="utf-8")

def log_print(msg: str = "", flush: bool = True):
    print(msg, flush=flush)
    try:
        log_file_handle.write(msg + "\n")
        if flush:
            log_file_handle.flush()
    except Exception:
        pass

def save_wav(path: str, pcm_data: Any, sample_rate: int = 24000):
    if isinstance(pcm_data, np.ndarray):
        if pcm_data.dtype == np.float32 or pcm_data.dtype == np.float64:
            pcm_data = np.clip(pcm_data, -1.0, 1.0)
            pcm_int16 = (pcm_data * 32767.0).astype(np.int16)
        else:
            pcm_int16 = pcm_data.astype(np.int16)
        raw_bytes = pcm_int16.tobytes()
    elif isinstance(pcm_data, (bytes, bytearray)):
        raw_bytes = bytes(pcm_data)
    elif isinstance(pcm_data, list):
        arr = np.array(pcm_data, dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        raw_bytes = (arr * 32767.0).astype(np.int16).tobytes()
    else:
        raise ValueError(f"Unsupported pcm format: {type(pcm_data)}")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)

# -------------------------------------------------------------
# TELEMETRY RECORDER
# -------------------------------------------------------------
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

            tok_gpu0 = 0
            tok_gpu1 = 0
            ceil_gpu0 = 33100
            ceil_gpu1 = 33100
            try:
                coord = getattr(self.orchestrator.cluster, "dynamic_coord", None)
                if coord:
                    with coord.lock:
                        tok_gpu0 = coord.active_tokens.get(0, 0)
                        tok_gpu1 = coord.active_tokens.get(1, 0)
                        ceil_gpu0 = getattr(coord, "gpu0_ceiling", getattr(coord, "ceiling_gpu0", 33100))
                        ceil_gpu1 = getattr(coord, "gpu1_ceiling", getattr(coord, "ceiling_gpu1_normal", 33100))
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

# -------------------------------------------------------------
# BENCHMARK RUNNER
# -------------------------------------------------------------
def run_1x15m_benchmark():
    log_print("=" * 80)
    log_print(">>> INITIALIZING DUAL TESLA T4 ENGINE (1x 15-MIN CFG=1.0 BENCHMARK)...")
    log_print(f">>> Persistent Live Log: {LOG_FILE_PATH}")
    log_print(f">>> Audio Output Directory: {AUDIO_DIR}")
    log_print("=" * 80)

    # 1. Start cluster with enable_asr=False -> Sizing: 33,100 tokens ceiling on both GPUs
    cluster = DualInstanceCluster(
        model_path=MODEL_PATH,
        lib_path=LIB_PATH,
        enable_q4_burst=False,
        enable_asr=False,
    )
    orchestrator = EngineOrchestrator(cluster=cluster)

    telemetry_csv = os.path.join(RESULTS_DIR, "telemetry_1x15m.csv")
    telemetry = SystemTelemetryLogger(orchestrator, telemetry_csv, interval_s=0.5)

    # 2. Prepare 1 request of 2,000 words (~15 minutes audio) with CFG = 1.0
    req_id = "req_1x15m_cfg1"
    target_words = 2000
    domain = "Astronomy & Deep Space Exploration"
    text = get_essay_script(0, target_words=target_words)
    actual_words = len(text.split())

    parcel = Parcel.tts_pipeline(
        script_text=text,
        instruction="Speak clearly, naturally, and with expressive prosody.",
        cfg_scale=1.0,
        follower_cfg_scale=1.0,
        temperature=0.60,
        enhance=False,
        master=False,
        srt=False,
        parcel_id=req_id,
    )

    log_print(f"\n>>> PREPARING SINGLE 15-MINUTE REQUEST:")
    log_print(f"  ID: {req_id}")
    log_print(f"  Domain: {domain}")
    log_print(f"  Words: {actual_words} words (~15.3 min audio)")
    log_print(f"  CFG Scale: 1.0 (Single-branch evaluation; no unconditional pass)")
    log_print(f"  Follower CFG Scale: 1.0")
    log_print(f"  ASR Enabled: False (Dynamic token ceiling = 33,100 on GPU 0 & GPU 1)")
    log_print(f"  Vocoder: Step 480 (Context 80) pipelined streaming")
    log_print("=" * 80)
    log_print(f">>> LAUNCHING REQUEST (t = 0.0s)...")
    log_print("=" * 80)

    telemetry.start()
    t_start = time.time()

    completed_result: Optional[ParcelResult] = None
    wall_s = 0.0

    try:
        # Dispatch in main thread and periodically print live telemetry
        def _dispatch_worker():
            nonlocal completed_result, wall_s
            t0 = time.time()
            completed_result = orchestrator.dispatch(parcel, timeout=1200.0)
            wall_s = time.time() - t0

        t_worker = threading.Thread(target=_dispatch_worker, daemon=True)
        t_worker.start()

        while t_worker.is_alive():
            t_worker.join(timeout=3.0)
            t_elapsed = time.time() - t_start
            recs = telemetry.records
            if recs:
                latest = recs[-1]
                log_print(
                    f"  [t={t_elapsed:5.1f}s] Active Tokens: GPU0={latest['tok_gpu0']:,} (ceil {latest['ceil_gpu0']:,}), "
                    f"GPU1={latest['tok_gpu1']:,} (ceil {latest['ceil_gpu1']:,}) | "
                    f"Util: GPU0={latest['gpu0_util']:3.0f}%, GPU1={latest['gpu1_util']:3.0f}% | "
                    f"Free VRAM: GPU0={latest['gpu0_free_mb']:.0f}MB, GPU1={latest['gpu1_free_mb']:.0f}MB | "
                    f"Pending Chunks: {latest['pending_chunks']}"
                )

    finally:
        total_wall_s = time.time() - t_start
        telemetry.stop()
        try:
            orchestrator.shutdown()
        except Exception:
            pass
        try:
            cluster.shutdown()
        except Exception:
            pass

    # Process and log results
    if completed_result is None or not completed_result.success:
        err = completed_result.error_message if completed_result else "Execution aborted"
        log_print(f"\n[-] BENCHMARK FAILED: {err}")
        return

    audio_pcm = getattr(completed_result, "audio_pcm", None)
    sr = getattr(completed_result, "sample_rate", 24000)
    dur = getattr(completed_result, "duration_s", 0.0)
    audio_dur = (len(audio_pcm) / float(sr)) if (audio_pcm is not None and len(audio_pcm) > 0) else dur
    rtf_speedup = audio_dur / max(0.001, wall_s)
    rtf_fraction = wall_s / max(0.001, audio_dur)

    wav_out = os.path.join(AUDIO_DIR, f"{req_id}.wav")
    save_wav(wav_out, audio_pcm, sr)
    wav_size_mb = os.path.getsize(wav_out) / (1024 * 1024)

    # Compute telemetry percentiles
    recs = telemetry.records
    u0 = [r["gpu0_util"] for r in recs] if recs else [0.0]
    u1 = [r["gpu1_util"] for r in recs] if recs else [0.0]
    f0 = [r["gpu0_free_mb"] for r in recs] if recs else [0.0]
    f1 = [r["gpu1_free_mb"] for r in recs] if recs else [0.0]

    summary = {
        "benchmark": "1x 15-Minute Isolated TTS Benchmark (CFG=1.0)",
        "request_id": req_id,
        "domain": domain,
        "words": actual_words,
        "cfg_scale": 1.0,
        "follower_cfg_scale": 1.0,
        "enable_asr": False,
        "audio_seconds": round(audio_dur, 2),
        "audio_minutes": round(audio_dur / 60.0, 2),
        "wall_seconds": round(wall_s, 2),
        "wall_minutes": round(wall_s / 60.0, 2),
        "speedup_xrealtime": round(rtf_speedup, 2),
        "rtf_fraction": round(rtf_fraction, 4),
        "wav_path": wav_out,
        "wav_size_mb": round(wav_size_mb, 2),
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
    }

    summary_file = os.path.join(RESULTS_DIR, "summary_1x15m.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log_print("\n" + "=" * 80)
    log_print(">>> 1x 15-MINUTE BENCHMARK COMPLETED SUCCESSFULLY!")
    log_print("=" * 80)
    log_print(f"  Total Words: {actual_words}")
    log_print(f"  Audio Generated: {audio_dur:.2f}s ({audio_dur/60.0:.2f} minutes)")
    log_print(f"  Wall Time: {wall_s:.2f}s ({wall_s/60.0:.2f} minutes)")
    log_print(f"  Speedup Multiplier: {rtf_speedup:.2f}x realtime (RTF fraction: {rtf_fraction:.4f})")
    log_print(f"  WAV File: {wav_out} ({wav_size_mb:.2f} MB)")
    log_print(f"  GPU 0 Compute: Mean={summary['gpu0_avg_compute_util_pct']:.2f}%, P50={summary['gpu0_p50_util_pct']:.2f}%, P90={summary['gpu0_p90_util_pct']:.2f}%, P99={summary['gpu0_p99_util_pct']:.2f}%")
    log_print(f"  GPU 1 Compute: Mean={summary['gpu1_avg_compute_util_pct']:.2f}%, P50={summary['gpu1_p50_util_pct']:.2f}%, P90={summary['gpu1_p90_util_pct']:.2f}%, P99={summary['gpu1_p99_util_pct']:.2f}%")
    log_print(f"  VRAM Min Free: GPU0={summary['min_free_vram_gpu0_mib']:.1f} MiB, GPU1={summary['min_free_vram_gpu1_mib']:.1f} MiB")
    log_print("=" * 80)

    # 3. Generate Analytical Visual Dashboard
    try:
        t_arr = np.array([r["elapsed_s"] for r in recs])
        u0_arr = np.array(u0)
        u1_arr = np.array(u1)
        f0_arr = np.array(f0)
        f1_arr = np.array(f1)
        tok0_arr = np.array([r["tok_gpu0"] for r in recs])
        tok1_arr = np.array([r["tok_gpu1"] for r in recs])

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Panel 1: Compute Utilization
        axs[0, 0].plot(t_arr, u0_arr, color="#1f77b4", label="GPU 0 Util %", alpha=0.85)
        axs[0, 0].plot(t_arr, u1_arr, color="#ff7f0e", label="GPU 1 Util %", alpha=0.85)
        axs[0, 0].axhline(np.mean(u0_arr), color="#1f77b4", linestyle="--", label=f"GPU 0 Mean: {np.mean(u0_arr):.1f}%")
        axs[0, 0].axhline(np.mean(u1_arr), color="#ff7f0e", linestyle="--", label=f"GPU 1 Mean: {np.mean(u1_arr):.1f}%")
        axs[0, 0].set_title("GPU Compute Utilization (CFG = 1.0)", fontsize=13, fontweight="bold")
        axs[0, 0].set_xlabel("Elapsed Time (seconds)")
        axs[0, 0].set_ylabel("Compute Utilization (%)")
        axs[0, 0].set_ylim(0, 105)
        axs[0, 0].legend(loc="lower right")

        # Panel 2: Free VRAM Headroom
        axs[0, 1].plot(t_arr, f0_arr, color="#2ca02c", label="GPU 0 Free VRAM (MiB)")
        axs[0, 1].plot(t_arr, f1_arr, color="#9467bd", label="GPU 1 Free VRAM (MiB)")
        axs[0, 1].axhline(500, color="red", linestyle=":", label="500 MiB Safety Margin")
        axs[0, 1].set_title("VRAM Free Headroom Over Time", fontsize=13, fontweight="bold")
        axs[0, 1].set_xlabel("Elapsed Time (seconds)")
        axs[0, 1].set_ylabel("Free VRAM (MiB)")
        axs[0, 1].legend(loc="lower left")

        # Panel 3: Active Tokens vs Ceiling
        axs[1, 0].plot(t_arr, tok0_arr, color="#1f77b4", label="GPU 0 Active Tokens")
        axs[1, 0].plot(t_arr, tok1_arr, color="#ff7f0e", label="GPU 1 Active Tokens")
        axs[1, 0].axhline(33100, color="black", linestyle="--", label="33,100 Token Ceiling")
        axs[1, 0].set_title("Active Token Timeline", fontsize=13, fontweight="bold")
        axs[1, 0].set_xlabel("Elapsed Time (seconds)")
        axs[1, 0].set_ylabel("Active Tokens")
        axs[1, 0].legend(loc="upper right")

        # Panel 4: Compute Percentile Distribution
        metrics = ["Mean", "P50", "P90", "P95", "P99"]
        vals_g0 = [np.mean(u0_arr), np.median(u0_arr), np.percentile(u0_arr, 90), np.percentile(u0_arr, 95), np.percentile(u0_arr, 99)]
        vals_g1 = [np.mean(u1_arr), np.median(u1_arr), np.percentile(u1_arr, 90), np.percentile(u1_arr, 95), np.percentile(u1_arr, 99)]
        x = np.arange(len(metrics))
        w = 0.35
        axs[1, 1].bar(x - w/2, vals_g0, width=w, color="#1f77b4", label="GPU 0")
        axs[1, 1].bar(x + w/2, vals_g1, width=w, color="#ff7f0e", label="GPU 1")
        axs[1, 1].set_title("Compute Percentiles (Mean, P50, P90, P95, P99)", fontsize=13, fontweight="bold")
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(metrics)
        axs[1, 1].set_ylabel("Utilization (%)")
        axs[1, 1].set_ylim(0, 105)
        for i, (v0, v1) in enumerate(zip(vals_g0, vals_g1)):
            axs[1, 1].text(i - w/2, v0 + 1, f"{v0:.1f}%", ha="center", fontsize=9)
            axs[1, 1].text(i + w/2, v1 + 1, f"{v1:.1f}%", ha="center", fontsize=9)
        axs[1, 1].legend(loc="lower right")

        plt.tight_layout()
        plot_out = os.path.join(RESULTS_DIR, "benchmark_1x15m_dashboard.png")
        fig.savefig(plot_out, dpi=180)
        plt.close(fig)
        log_print(f">>> Generated visual dashboard: {plot_out}")
    except Exception as pe:
        log_print(f"[-] Plotting exception caught: {pe}")

if __name__ == "__main__":
    run_1x15m_benchmark()
