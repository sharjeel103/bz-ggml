"""
5x 20-Minute Concurrent TTS Saturation Benchmark on Dual Tesla T4 GPUs.
Evaluates continuous sustained decoding under heavy load:
  - 5 simultaneous requests of ~3,000 words each (~20 min audio each, ~100 min total).
  - Authentic human English text across 5 distinct domains (Astronomy, Archaeology, Oceanography, History of Computing, Flight).
  - Req 1 & 2: cfg_scale = 5.0 (Extreme CFG)
  - Req 3, 4, 5: cfg_scale = 1.0 (Zero-unconditional bypass)
  - Audio saved as 24kHz 16-bit WAV files into /kaggle/working/benchmark_results/audio/
  - Continuous persistent logging to /kaggle/working/benchmark_results/benchmark_live.log
  - High-frequency continuous telemetry (500ms): GPU compute util %, VRAM headroom, active tokens.
"""

import sys, os, time, math, threading, json, subprocess, wave
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
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
LOG_FILE_PATH = os.path.join(RESULTS_DIR, "benchmark_live.log")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------------------
# DUAL TEE LOGGER (CONSOLE + PERSISTENT DISK LOG)
# -------------------------------------------------------------
log_file_handle = open(LOG_FILE_PATH, "a", encoding="utf-8")

def log_print(msg: str = "", flush: bool = True):
    """Prints to stdout and writes to persistent log file on Kaggle disk."""
    sys.stdout.write(msg + "\n")
    if flush:
        sys.stdout.flush()
    try:
        log_file_handle.write(msg + "\n")
        if flush:
            log_file_handle.flush()
    except Exception:
        pass

def save_wav(filepath: str, pcm_data: np.ndarray, sample_rate: int = 24000):
    """Saves numpy PCM array to standard 16-bit mono WAV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if pcm_data.dtype in (np.float32, np.float64):
        int16_data = (np.clip(pcm_data, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        int16_data = pcm_data.astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

# -------------------------------------------------------------
# HIGH-PRECISION CONTINUOUS TELEMETRY MONITOR
# -------------------------------------------------------------
class SystemTelemetryLogger:
    def __init__(self, orchestrator: EngineOrchestrator, log_path: str, interval_s: float = 0.5):
        self.orchestrator = orchestrator
        self.log_path = log_path
        self.interval_s = interval_s
        self.records: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._csv_file = open(self.log_path, "w", encoding="utf-8")
        self._csv_headers_written = False

    def start(self):
        self._stop_event.clear()
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

    def _get_nvidia_smi(self) -> Dict[str, Any]:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            lines = res.stdout.strip().split("\n")
            stats = {}
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
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

            # Continuous disk write with flush so telemetry never vanishes
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
def run_5x20m_saturation_benchmark():
    log_print("=" * 80)
    log_print(">>> INITIALIZING DUAL TESLA T4 PRODUCTION ENGINE...")
    log_print(f">>> Persistent Live Log: {LOG_FILE_PATH}")
    log_print(f">>> Audio Output Directory: {AUDIO_DIR}")
    log_print("=" * 80)

    cluster = DualInstanceCluster(
        model_path=MODEL_PATH,
        lib_path=LIB_PATH,
        enable_q4_burst=False,
        enable_asr=False,
    )
    orchestrator = EngineOrchestrator(cluster=cluster)

    telemetry_csv = os.path.join(RESULTS_DIR, "telemetry_5x20m.csv")
    telemetry = SystemTelemetryLogger(orchestrator, telemetry_csv, interval_s=0.5)

    # Prepare 5 requests of ~3,000 words each across 5 authentic domains
    # User requirement: cfg to 1 for 3 files and cfg 5 to 2
    configs = [
        {"id": "req_20m_01_cfg5", "cfg_scale": 5.0, "words": 3000, "domain": "Astronomy: Cosmic Horizon & Voyagers", "essay_id": 0},
        {"id": "req_20m_02_cfg5", "cfg_scale": 5.0, "words": 3000, "domain": "Archaeology: Architecture of Forgotten Realms", "essay_id": 1},
        {"id": "req_20m_03_cfg1", "cfg_scale": 1.0, "words": 3000, "domain": "Oceanography: Deep Sea Midnight Zone", "essay_id": 2},
        {"id": "req_20m_04_cfg1", "cfg_scale": 1.0, "words": 3000, "domain": "Computation: Clay Tablets to Neural Nets", "essay_id": 3},
        {"id": "req_20m_05_cfg1", "cfg_scale": 1.0, "words": 3000, "domain": "Aviation: Physics & Art of Flight", "essay_id": 4},
    ]

    log_print("\n>>> PREPARING 5 CONCURRENT 20-MINUTE AUTHENTIC HUMAN TEXT REQUESTS:")
    parcels: List[Parcel] = []
    for c in configs:
        text = get_essay_script(c["essay_id"], target_words=c["words"])
        p = Parcel.tts_pipeline(
            script_text=text,
            instruction="Speak clearly, naturally, and with expressive prosody.",
            cfg_scale=c["cfg_scale"],
            temperature=0.60,
            enhance=False,
            master=False,
            srt=False,
            parcel_id=c["id"],
        )
        parcels.append(p)
        actual_words = len(text.split())
        log_print(f"  [{c['id']}] {c['domain']} | {actual_words} words, CFG = {c['cfg_scale']}, ~20 min audio.")

    log_print("\n" + "=" * 80)
    log_print(">>> LAUNCHING ALL 5 REQUESTS SIMULTANEOUSLY AT ONCE (t = 0.0s)...")
    log_print("=" * 80)

    telemetry.start()
    t_start = time.time()

    completed_results: Dict[str, ParcelResult] = {}
    completed_timings: Dict[str, float] = {}

    def _save_incremental_summary():
        try:
            cur_summary = {
                "in_progress": True,
                "elapsed_s": round(time.time() - t_start, 2),
                "completed_count": len(completed_results),
                "total_count": len(configs),
                "completed": [
                    {
                        "id": pid,
                        "success": res.success,
                        "audio_seconds": round(len(res.audio_pcm) / float(res.sample_rate), 2) if (res.audio_pcm is not None and len(res.audio_pcm) > 0) else 0.0,
                        "wall_s": round(completed_timings.get(pid, 0.0), 2),
                    }
                    for pid, res in completed_results.items()
                ]
            }
            with open(os.path.join(RESULTS_DIR, "summary_5x20m.json"), "w") as f:
                json.dump(cur_summary, f, indent=2)
        except Exception:
            pass

    def _run_single(parcel: Parcel):
        pid = parcel.parcel_id
        t0 = time.time()
        try:
            res = orchestrator.dispatch(parcel, timeout=2400.0)
            wall_s = time.time() - t0
            completed_timings[pid] = wall_s
            completed_results[pid] = res

            audio_pcm = getattr(res, "audio_pcm", None)
            sr = getattr(res, "sample_rate", 24000)
            dur = getattr(res, "duration_s", 0.0)
            audio_dur = (len(audio_pcm) / float(sr)) if (audio_pcm is not None and len(audio_pcm) > 0) else dur
            rtf = audio_dur / max(0.001, wall_s)
            status_str = "SUCCESS" if res.success else f"FAILED ({res.error_message})"
            
            # Save audio immediately to Kaggle output directory
            if res.success and audio_pcm is not None and len(audio_pcm) > 0:
                wav_out = os.path.join(AUDIO_DIR, f"{pid}.wav")
                save_wav(wav_out, audio_pcm, sr)
                size_mb = os.path.getsize(wav_out) / (1024 * 1024)
                log_print(f"\n[+] FINISHED [{pid}]: {status_str} | Wall: {wall_s:.1f}s | Audio: {audio_dur:.1f}s ({audio_dur/60:.2f}m) | RTF: {rtf:.2f}x | Saved: {wav_out} ({size_mb:.1f} MB)")
            else:
                log_print(f"\n[+] FINISHED [{pid}]: {status_str} | Wall: {wall_s:.1f}s | Audio: {audio_dur:.1f}s | RTF: {rtf:.2f}x")

            _save_incremental_summary()
            return res
        except Exception as e:
            wall_s = time.time() - t0
            completed_timings[pid] = wall_s
            err_res = ParcelResult(parcel_id=pid, success=False, error_message=str(e))
            completed_results[pid] = err_res
            log_print(f"\n[-] ERROR in [{pid}]: {e}")
            _save_incremental_summary()
            return err_res

    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_run_single, p) for p in parcels]
            while any(not f.done() for f in futures):
                time.sleep(5.0)
                t_elapsed = time.time() - t_start
                recs = telemetry.records
                if recs:
                    latest = recs[-1]
                    log_print(f"  [t={t_elapsed:5.1f}s] Active Tokens: GPU0={latest['tok_gpu0']:,} (ceil {latest['ceil_gpu0']:,}), GPU1={latest['tok_gpu1']:,} (ceil {latest['ceil_gpu1']:,}) | Util: GPU0={latest['gpu0_util']:3.0f}%, GPU1={latest['gpu1_util']:3.0f}% | VRAM Free: GPU0={latest['gpu0_free_mb']:.0f}MB, GPU1={latest['gpu1_free_mb']:.0f}MB | Pending Chunks: {latest['pending_chunks']}")

            for f in futures:
                try:
                    f.result()
                except Exception as fe:
                    log_print(f"[-] Future exception caught: {fe}")

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

    # -------------------------------------------------------------
    # METRICS CALCULATION
    # -------------------------------------------------------------
    log_print("\n" + "=" * 80)
    log_print(">>> 5x 20-MINUTE BENCHMARK EXECUTION SUMMARY:")
    log_print("=" * 80)

    total_audio_s = 0.0
    request_summaries = []
    for c in configs:
        pid = c["id"]
        res = completed_results.get(pid)
        wall_s = completed_timings.get(pid, total_wall_s)
        if res:
            audio_pcm = getattr(res, "audio_pcm", None)
            sr = getattr(res, "sample_rate", 24000)
            dur = getattr(res, "duration_s", 0.0)
            audio_s = (len(audio_pcm) / float(sr)) if (audio_pcm is not None and len(audio_pcm) > 0) else dur
            total_audio_s += audio_s
            req_rtf = audio_s / max(0.001, wall_s)
            wav_path = os.path.join(AUDIO_DIR, f"{pid}.wav")
            wav_exists = os.path.exists(wav_path)
            wav_size_mb = (os.path.getsize(wav_path) / (1024 * 1024)) if wav_exists else 0.0

            req_sum = {
                "id": pid,
                "cfg_scale": c["cfg_scale"],
                "domain": c["domain"],
                "words": c["words"],
                "success": res.success,
                "wall_time_s": round(wall_s, 2),
                "audio_duration_s": round(audio_s, 2),
                "audio_duration_m": round(audio_s / 60.0, 2),
                "rtf_realtime_factor": round(req_rtf, 2),
                "x_realtime": round(req_rtf, 2),
                "rtf_fraction": round(1.0 / max(0.001, req_rtf), 4),
                "wav_path": wav_path,
                "wav_size_mb": round(wav_size_mb, 2),
                "error": res.error_message,
            }
        else:
            req_sum = {
                "id": pid,
                "cfg_scale": c["cfg_scale"],
                "domain": c["domain"],
                "words": c["words"],
                "success": False,
                "error": "Timed out or missing",
            }
        request_summaries.append(req_sum)

    aggregate_rtf = total_audio_s / max(0.001, total_wall_s)

    # Compute saturation stats from telemetry
    recs = telemetry.records
    sat_gpu0_s = sum(0.5 for r in recs if r["gpu0_util"] >= 90.0)
    sat_gpu1_s = sum(0.5 for r in recs if r["gpu1_util"] >= 90.0)
    sat_both_s = sum(0.5 for r in recs if r["gpu0_util"] >= 90.0 and r["gpu1_util"] >= 90.0)
    avg_gpu0_util = np.mean([r["gpu0_util"] for r in recs]) if recs else 0.0
    avg_gpu1_util = np.mean([r["gpu1_util"] for r in recs]) if recs else 0.0
    min_gpu0_free = min([r["gpu0_free_mb"] for r in recs]) if recs else 0.0
    min_gpu1_free = min([r["gpu1_free_mb"] for r in recs]) if recs else 0.0
    max_tok_gpu0 = max([r["tok_gpu0"] for r in recs]) if recs else 0
    max_tok_gpu1 = max([r["tok_gpu1"] for r in recs]) if recs else 0

    log_print(f"Total Requests:              5 (Simultaneous)")
    log_print(f"Total Words Synthesized:     15,000 words (Authentic human prose)")
    log_print(f"Total Audio Generated:       {total_audio_s:.1f} s ({total_audio_s / 60.0:.2f} minutes)")
    log_print(f"Total Wall-Clock Time:       {total_wall_s:.1f} s ({total_wall_s / 60.0:.2f} minutes)")
    log_print(f"Aggregate Cluster RTF:       {aggregate_rtf:.2f}x Real-Time ({1.0 / aggregate_rtf:.4f} s/s)")
    log_print(f"Average GPU Compute Util:    GPU 0: {avg_gpu0_util:.1f}%, GPU 1: {avg_gpu1_util:.1f}%")
    log_print(f"Sustained Saturation (>=90%): GPU 0: {sat_gpu0_s:.1f}s, GPU 1: {sat_gpu1_s:.1f}s, Both: {sat_both_s:.1f}s")
    log_print(f"Peak Concurrent Tokens:      GPU 0: {max_tok_gpu0:,} / 36,900 | GPU 1: {max_tok_gpu1:,} / 36,900")
    log_print(f"Minimum Free VRAM Headroom:  GPU 0: {min_gpu0_free:.0f} MiB | GPU 1: {min_gpu1_free:.0f} MiB (Target >= 500 MiB)")

    log_print("\nINDIVIDUAL REQUEST RESULTS:")
    for rs in request_summaries:
        status_str = "SUCCESS" if rs.get("success") else "FAILED"
        log_print(f"  [{rs['id']}] CFG={rs['cfg_scale']} | {rs['domain']}: {status_str} | Wall: {rs.get('wall_time_s',0):.1f}s | Audio: {rs.get('audio_duration_m',0):.2f}m | RTF: {rs.get('rtf_realtime_factor',0):.2f}x | WAV: {rs.get('wav_size_mb',0)} MB")

    summary_data = {
        "benchmark": "5x 20-Minute Concurrent TTS Saturation",
        "total_requests": 5,
        "total_words": 15000,
        "total_audio_seconds": round(total_audio_s, 2),
        "total_audio_minutes": round(total_audio_s / 60.0, 2),
        "total_wall_seconds": round(total_wall_s, 2),
        "total_wall_minutes": round(total_wall_s / 60.0, 2),
        "aggregate_cluster_rtf": round(aggregate_rtf, 2),
        "aggregate_cluster_xrealtime": round(aggregate_rtf, 2),
        "gpu0_avg_compute_util_pct": round(float(avg_gpu0_util), 2),
        "gpu1_avg_compute_util_pct": round(float(avg_gpu1_util), 2),
        "gpu0_saturation_ge_90pct_seconds": round(sat_gpu0_s, 2),
        "gpu1_saturation_ge_90pct_seconds": round(sat_gpu1_s, 2),
        "both_gpus_saturation_ge_90pct_seconds": round(sat_both_s, 2),
        "min_free_vram_gpu0_mib": round(float(min_gpu0_free), 2),
        "min_free_vram_gpu1_mib": round(float(min_gpu1_free), 2),
        "peak_tokens_gpu0": int(max_tok_gpu0),
        "peak_tokens_gpu1": int(max_tok_gpu1),
        "requests": request_summaries,
    }

    summary_path = os.path.join(RESULTS_DIR, "summary_5x20m.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # -------------------------------------------------------------
    # PLOT GENERATION
    # -------------------------------------------------------------
    if recs:
        times = [r["elapsed_s"] for r in recs]

        # 1. GPU Compute Utilization
        plt.figure(figsize=(12, 5))
        plt.plot(times, [r["gpu0_util"] for r in recs], label="GPU 0 Compute % (T4)", color="#1f77b4", linewidth=1.8)
        plt.plot(times, [r["gpu1_util"] for r in recs], label="GPU 1 Compute % (T4)", color="#ff7f0e", linewidth=1.8)
        plt.axhline(90, color="red", linestyle="--", alpha=0.7, label="90% Saturation Threshold")
        plt.title(f"Dual Tesla T4 GPU Compute Saturation (5x 20-min TTS Requests | Aggregate RTF: {aggregate_rtf:.1f}x)", fontsize=13, fontweight="bold")
        plt.xlabel("Elapsed Time (seconds)", fontsize=11)
        plt.ylabel("Compute Utilization (%)", fontsize=11)
        plt.ylim(0, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "gpu_compute_5x20m.png"), dpi=200)
        plt.close()

        # 2. VRAM Saturation & Headroom
        plt.figure(figsize=(12, 5))
        plt.plot(times, [r["gpu0_used_mb"] for r in recs], label="GPU 0 VRAM Used", color="#1f77b4", linewidth=1.8)
        plt.plot(times, [r["gpu1_used_mb"] for r in recs], label="GPU 1 VRAM Used", color="#ff7f0e", linewidth=1.8)
        plt.axhline(15360 - 550, color="red", linestyle="--", alpha=0.7, label="550 MiB Free Target (14,810 MiB)")
        plt.title("VRAM Allocation & Strict 500-600 MiB Free Margin (Zero OOM Safety)", fontsize=13, fontweight="bold")
        plt.xlabel("Elapsed Time (seconds)", fontsize=11)
        plt.ylabel("VRAM Allocated (MiB)", fontsize=11)
        plt.ylim(10000, 15500)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "vram_saturation_5x20m.png"), dpi=200)
        plt.close()

        # 3. Active Tokens Concurrency Timeline
        plt.figure(figsize=(12, 5))
        plt.plot(times, [r["tok_gpu0"] for r in recs], label="GPU 0 Active Tokens", color="#1f77b4", linewidth=1.8)
        plt.plot(times, [r["tok_gpu1"] for r in recs], label="GPU 1 Active Tokens", color="#ff7f0e", linewidth=1.8)
        plt.axhline(33100, color="#1f77b4", linestyle=":", alpha=0.8, label="GPU 0 Ceiling (33,100)")
        plt.axhline(33100, color="#ff7f0e", linestyle=":", alpha=0.8, label="GPU 1 Ceiling (33,100)")
        plt.title("Token Capacity Coordinator: Active In-Flight Tokens", fontsize=13, fontweight="bold")
        plt.xlabel("Elapsed Time (seconds)", fontsize=11)
        plt.ylabel("In-Flight Tokens", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "tokens_timeline_5x20m.png"), dpi=200)
        plt.close()

    log_print(f"\n>>> Benchmark artifacts successfully generated in {RESULTS_DIR}:")
    log_print(f"    - live log: {LOG_FILE_PATH}")
    log_print(f"    - telemetry: {telemetry_csv}")
    log_print(f"    - summary: {summary_path}")
    log_print(f"    - audio files in: {AUDIO_DIR}/")
    log_print(f"    - gpu_compute_5x20m.png")
    log_print(f"    - vram_saturation_5x20m.png")
    log_print(f"    - tokens_timeline_5x20m.png")
    log_print("=" * 80)

if __name__ == "__main__":
    run_5x20m_saturation_benchmark()
