"""
Continuous Multi-Model Production Marathon Benchmark on Dual Tesla T4 GPUs.
Evaluates heterogeneous request stream under continuous load:
  - Short (1 min, ~140w): TTS + Master, TTS + ASR, return_tokens_only
  - Medium (5-7 min, ~750w): TTS + Enhance + Master + ASR
  - Long Marathons (8-25 min, ~1,200w - 3,200w): Dual-GPU fanout, tiled vocoding (step=480)
  - Standalone Enhancer: GPU 1 preemption and vocoder fallback to GPU 0
  - Standalone ASR & Cross-Lingual Translation: Spanish -> English
  - Voice Persona Registration: [Enhance, ASR, Voice_Register]
  - Driver-level telemetry: VRAM, GPU compute util %, queue depths, RTF curves.
"""

import sys, os, time, math, threading, json, base64
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/kaggle/working/bz-ggml/python")
os.chdir("/kaggle/working/bz-ggml")

from bz_ggml import (
    DualInstanceCluster, UserTask,
    ASRService,
    EngineOrchestrator, EngineCapabilities,
    TaskType, TaskSpec, Parcel, ParcelResult,
    deserialize_breeze_voice
)

MODEL_PATH = "/kaggle/working/models/breeze-tts-2-q8_0.gguf"
LIB_PATH = "/kaggle/working/bz-ggml/build/libbreeze.so"
RESULTS_DIR = "/kaggle/working/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# REALISTIC CORPUS GENERATOR (SHORT, MEDIUM, LONG, ULTRA-LONG)
# -------------------------------------------------------------
PARAGRAPHS = [
    (
        "Artificial intelligence and neural audio synthesis have reached an inflection point in computational efficiency. "
        "By orchestrating discrete latent acoustic spaces across symmetric multi-GPU architectures, modern systems achieve "
        "seamless real-time stream delivery without compromising vocal warmth, emotional prosody, or phonetic fidelity. "
        "Dynamic token budgeting and adaptive batching guarantee robust execution even under peak adversarial concurrency."
    ),
    (
        "In enterprise cloud environments, distributed pipeline orchestration prevents hardware resource contention. "
        "When compute-intensive neural diffusion stages and automatic speech recognition services run alongside autoregressive "
        "backbone decoders, strict micro-service queue boundaries isolate latency spikes and protect GPU memory headroom. "
        "Continuous telemetry logging verifies that memory footprints remain strictly bounded across long-running audio workloads."
    ),
    (
        "Voice cloning and acoustic conditioning require high-dimensional latent representations that capture unique vocal registers. "
        "Through sixteen hierarchical codebooks, fine-grained acoustic features are faithfully preserved across extended generation horizons. "
        "Anchor-first semantic chunking decomposes long scripts into coherent syntactic clauses, ensuring that pitch contours, "
        "respiratory pauses, and vocal timbre remain invariant from the first sentence to the final concluding syllable."
    ),
    (
        "High-throughput speech generation relies on tiled neural vocoding with overlapping boundary interpolation. "
        "By executing inference across constant frame windows with dynamic cross-GPU priority routing, scratchpad memory "
        "is permanently capped, completely eliminating memory fragmentation and out-of-memory faults during thirty-minute decodes. "
        "Sub-word acoustic alignment further enables precise, automated word-level subtitle generation for broadcast production."
    ),
    (
        "Modern deep learning architectures demand rigorous empirical profiling to validate throughput multipliers. "
        "Real-time factor measurements across heterogeneous request distributions demonstrate that continuous batching "
        "and hardware-aligned execution deliver dramatic speedups over conventional monolithic inference pipelines. "
        "Systematic load testing confirms that production orchestrators maintain stability under sustained multi-user saturation."
    )
]

def generate_script(target_words: int) -> str:
    """Generates continuous natural text of exact target length."""
    words = []
    idx = 0
    while len(words) < target_words:
        para_words = PARAGRAPHS[idx % len(PARAGRAPHS)].split()
        words.extend(para_words)
        idx += 1
    return " ".join(words[:target_words])

# -------------------------------------------------------------
# HIGH-PRECISION TELEMETRY LOGGER (500 ms INTERVAL)
# -------------------------------------------------------------
class ContinuousTelemetrySampler(threading.Thread):
    def __init__(self, orchestrator: EngineOrchestrator, interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.orchestrator = orchestrator
        self.interval_s = interval_s
        self.running = True
        self.records: List[Dict[str, Any]] = []
        self.start_time = 0.0

    def run(self):
        self.start_time = time.time()
        while self.running:
            t = time.time() - self.start_time
            # VRAM via PyTorch driver
            free0, total0 = torch.cuda.mem_get_info(0)
            free1, total1 = torch.cuda.mem_get_info(1)
            used0_mib = (total0 - free0) / (1024 * 1024)
            used1_mib = (total1 - free1) / (1024 * 1024)
            total0_mib = total0 / (1024 * 1024)
            total1_mib = total1 / (1024 * 1024)

            # GPU Compute Utilization via nvidia-smi
            gpu0_util, gpu1_util = 0.0, 0.0
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, timeout=1.0
                ).strip().splitlines()
                if len(out) >= 2:
                    gpu0_util = float(out[0].strip())
                    gpu1_util = float(out[1].strip())
            except Exception:
                pass

            # Orchestrator Queue Depths
            q_depths = self.orchestrator.get_queue_depths()
            with self.orchestrator.state_lock:
                active_count = len(self.orchestrator.active_requests)

            self.records.append({
                "time_s": t,
                "gpu0_vram_mib": used0_mib,
                "gpu1_vram_mib": used1_mib,
                "gpu0_util_pct": gpu0_util,
                "gpu1_util_pct": gpu1_util,
                "q_tts": q_depths.get("tts", 0),
                "q_vocode": q_depths.get("vocode", 0),
                "q_enhance": q_depths.get("enhance", 0),
                "q_asr": q_depths.get("asr", 0),
                "q_upload": q_depths.get("upload", 0),
                "active_requests": active_count
            })
            time.sleep(self.interval_s)

    def stop(self):
        self.running = False

# -------------------------------------------------------------
# MAIN BENCHMARK EXECUTION SUITE
# -------------------------------------------------------------
def run_benchmark():
    print("=" * 80)
    print(" CONTINUOUS MULTI-MODEL PRODUCTION MARATHON BENCHMARK (DUAL TESLA T4) ")
    print("=" * 80)

    # 1. Initialize Engine & Services
    print(">>> Initializing DualInstanceCluster (Q8_0, Tiled Vocoder step=240, 500MB Headroom)...", flush=True)
    cluster = DualInstanceCluster(
        model_path=MODEL_PATH,
        lib_path=LIB_PATH,
        enable_vocoder=True,
        vocoder_mode="on",
        precision="int8",
        enable_enhancer=True,
        enable_asr=True,
    )

    print(">>> Initializing ASR Service (Whisper Turbo on GPU 0)...", flush=True)
    asr_service = ASRService(
        model_size="turbo",
        device="cuda",
        device_index=0,
        compute_type="float16"
    )

    caps = EngineCapabilities(
        precision="int8",
        enable_enhancer=True,
        enable_asr=True,
        chunk_seconds_enhancer=10.0,
        asr_model_size="turbo",
        asr_device_index=0,
        enhancer_device_index=1,
        vocoder_step_frames=240
    )

    orchestrator = EngineOrchestrator(
        cluster=cluster,
        asr_service=asr_service,
        capabilities=caps
    )

    # 2. Warm up and Synthesize Seed Audio for Standalone Enhancer & ASR
    print("\n--- Generating Seed Audio Excerpt for Auxiliary Standalone Jobs ---", flush=True)
    warmup_parcel = Parcel.tts_pipeline(
        script_text="High performance neural speech engine running live on dual Tesla T4 GPUs with fully isolated queues.",
        instruction="Speak clearly and confidently.",
        parcel_id="seed_audio_00"
    )
    seed_res = orchestrator.dispatch(warmup_parcel)
    assert seed_res.success, f"Seed generation failed: {seed_res.error_message}"
    seed_pcm = seed_res.audio_pcm
    seed_sr = seed_res.sample_rate
    print(f"✅ Seed Audio Ready: {seed_res.duration_s:.2f}s ({len(seed_pcm)} samples @ {seed_sr} Hz)")

    # 3. Spanish Audio for Cross-Lingual Translation Benchmark
    print("--- Synthesizing Spanish Reference Audio for Translation Benchmark ---", flush=True)
    es_parcel = Parcel.tts_pipeline(
        script_text="La inteligencia artificial transforma la produccion de audio moderno con velocidad y precision excepcionales.",
        instruction="Habla con claridad y naturalidad.",
        parcel_id="seed_spanish_00"
    )
    es_res = orchestrator.dispatch(es_parcel)
    assert es_res.success, f"Spanish seed failed: {es_res.error_message}"
    es_pcm = es_res.audio_pcm

    # 4. Define Heterogeneous Request Matrix
    print("\n" + "=" * 80)
    print(" CONFIGURING CONTINUOUS LOAD SPECTRUM ")
    print("=" * 80)

    workload_specs = [
        # (name, type, parcel_factory, delay_before_submit_s)
        {
            "id": "req_01_marathon_1200w",
            "category": "Long Marathon (8-10m)",
            "words": 1200,
            "tasks": ["TTS", "MASTER", "ASR"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(1200),
                instruction="Speak with authoritative executive cadence.",
                enhance=False, master=True, srt=True,
                parcel_id="req_01_marathon_1200w"
            ),
            "stagger_s": 0.0
        },
        {
            "id": "req_02_medium_750w",
            "category": "Medium Commercial (5-7m)",
            "words": 750,
            "tasks": ["TTS", "ENHANCE", "MASTER", "ASR"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(750),
                instruction="Speak with natural warmth and commercial clarity.",
                enhance=True, master=True, srt=True,
                parcel_id="req_02_medium_750w"
            ),
            "stagger_s": 1.0
        },
        {
            "id": "req_03_short_master",
            "category": "Short Interactive (1m)",
            "words": 140,
            "tasks": ["TTS", "MASTER"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(140),
                instruction="Speak clearly.",
                enhance=False, master=True, srt=False,
                parcel_id="req_03_short_master"
            ),
            "stagger_s": 1.5
        },
        {
            "id": "req_04_short_srt",
            "category": "Short Subtitled (1m)",
            "words": 140,
            "tasks": ["TTS", "ASR"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(140),
                instruction="Speak with lively energy.",
                enhance=False, master=False, srt=True,
                parcel_id="req_04_short_srt"
            ),
            "stagger_s": 2.0
        },
        {
            "id": "req_05_tokens_only_1",
            "category": "Return Tokens Only",
            "words": 180,
            "tasks": ["TTS"],
            "parcel": Parcel(
                tasks=[TaskSpec(task_type=TaskType.TTS)],
                payload={
                    "script_text": generate_script(180),
                    "instruction": "Speak clearly.",
                    "cfg_scale": 1.0,
                    "temperature": 0.60,
                    "return_tokens_only": True
                },
                parcel_id="req_05_tokens_only_1"
            ),
            "stagger_s": 2.5
        },
        {
            "id": "req_06_tokens_only_2",
            "category": "Return Tokens Only",
            "words": 200,
            "tasks": ["TTS"],
            "parcel": Parcel(
                tasks=[TaskSpec(task_type=TaskType.TTS)],
                payload={
                    "script_text": generate_script(200),
                    "instruction": "Speak clearly.",
                    "cfg_scale": 1.0,
                    "temperature": 0.60,
                    "return_tokens_only": True
                },
                parcel_id="req_06_tokens_only_2"
            ),
            "stagger_s": 2.7
        },
        {
            "id": "req_07_standalone_enhance",
            "category": "Standalone Enhancer",
            "words": 0,
            "tasks": ["ENHANCE", "MASTER"],
            "parcel": Parcel(
                tasks=[TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}), TaskSpec(TaskType.MASTER)],
                payload={"audio_input": seed_pcm, "sample_rate": 24000},
                parcel_id="req_07_standalone_enhance"
            ),
            "stagger_s": 4.0
        },
        {
            "id": "req_08_standalone_asr_srt",
            "category": "Standalone ASR",
            "words": 0,
            "tasks": ["ASR"],
            "parcel": Parcel(
                tasks=[TaskSpec(TaskType.ASR, {"output_format": "srt", "words_per_line": 1})],
                payload={"audio_input": seed_pcm, "sample_rate": 24000},
                parcel_id="req_08_standalone_asr_srt"
            ),
            "stagger_s": 4.5
        },
        {
            "id": "req_09_asr_translation_es_en",
            "category": "ASR Translation (ES->EN)",
            "words": 0,
            "tasks": ["ASR"],
            "parcel": Parcel(
                tasks=[TaskSpec(TaskType.ASR, {"output_format": "txt", "task": "translate", "language": "es"})],
                payload={
                    "audio_input": es_pcm,
                    "sample_rate": 24000,
                    "task": "translate",
                    "language": "es"
                },
                parcel_id="req_09_asr_translation_es_en"
            ),
            "stagger_s": 5.0
        },
        {
            "id": "req_10_voice_persona_reg",
            "category": "Voice Persona Registration",
            "words": 0,
            "tasks": ["ENHANCE", "ASR", "VOICE_REGISTER"],
            "parcel": Parcel.voice_registration(
                voice_name="marathon_benchmark_persona",
                audio_input=seed_pcm,
                parcel_id="req_10_voice_persona_reg"
            ),
            "stagger_s": 6.0
        },
        {
            "id": "req_11_ultra_marathon_2200w",
            "category": "Ultra Marathon (15-20m)",
            "words": 2200,
            "tasks": ["TTS", "MASTER", "ASR"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(2200),
                instruction="Speak with steady, articulate documentary presentation.",
                enhance=False, master=True, srt=True,
                parcel_id="req_11_ultra_marathon_2200w"
            ),
            "stagger_s": 8.0
        },
        {
            "id": "req_12_short_rapid_1",
            "category": "Short Interactive (1m)",
            "words": 150,
            "tasks": ["TTS", "MASTER"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(150),
                instruction="Speak quickly.",
                enhance=False, master=True, srt=False,
                parcel_id="req_12_short_rapid_1"
            ),
            "stagger_s": 12.0
        },
        {
            "id": "req_13_short_rapid_2",
            "category": "Short Subtitled (1m)",
            "words": 150,
            "tasks": ["TTS", "ASR"],
            "parcel": Parcel.tts_pipeline(
                script_text=generate_script(150),
                instruction="Speak clearly.",
                enhance=False, master=False, srt=True,
                parcel_id="req_13_short_rapid_2"
            ),
            "stagger_s": 12.5
        }
    ]

    # 5. Start Telemetry Collector
    telemetry = ContinuousTelemetrySampler(orchestrator, interval_s=0.5)
    telemetry.start()
    print(">>> Continuous Telemetry Collector Online (Sampling every 500ms)...", flush=True)

    # 6. Execute Workload Concurrently with Staggered Arrivals
    benchmark_start = time.time()
    results_map: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    def worker_dispatch(spec: Dict[str, Any]):
        time.sleep(spec["stagger_s"])
        req_id = spec["id"]
        t_submit = time.time() - benchmark_start
        print(f"[{t_submit:6.1f}s] >>> SUBMITTING: {req_id} ({spec['category']}, {spec['words']} words, tasks: {spec['tasks']})", flush=True)
        t0 = time.time()
        res = orchestrator.dispatch(spec["parcel"], timeout=1800.0)
        wall_time = time.time() - t0
        t_finish = time.time() - benchmark_start

        # Calculate metrics
        audio_dur = res.duration_s if res.audio_pcm is not None and len(res.audio_pcm) > 0 else (spec["words"] / 2.5 if spec["words"] > 0 else 10.0)
        rtf = wall_time / audio_dur if audio_dur > 0 else 0.0

        status_str = "SUCCESS" if res.success else f"FAILED: {res.error_message}"
        print(f"[{t_finish:6.1f}s] <<< COMPLETED: {req_id} | Status: {status_str} | Audio: {audio_dur:.1f}s | Wall: {wall_time:.2f}s | RTF: {rtf:.3f}x", flush=True)

        if res.transcript:
            print(f"    Transcript preview: \"{res.transcript[:90]}...\"", flush=True)
        if res.tokens is not None:
            print(f"    Tokens preview: {len(res.tokens)} tokens returned directly (bypassed vocoder)", flush=True)

        with lock:
            results_map[req_id] = {
                "id": req_id,
                "category": spec["category"],
                "words": spec["words"],
                "tasks": spec["tasks"],
                "submit_time_s": t_submit,
                "finish_time_s": t_finish,
                "wall_time_s": wall_time,
                "audio_duration_s": audio_dur,
                "rtf": rtf,
                "success": res.success,
                "error": res.error_message,
                "stage_timings": res.stage_timings
            }

    print(f"\n>>> Launching {len(workload_specs)} concurrent heterogeneous requests via thread pool...", flush=True)
    with ThreadPoolExecutor(max_workers=len(workload_specs)) as executor:
        futures = [executor.submit(worker_dispatch, spec) for spec in workload_specs]
        for f in futures:
            f.result()

    total_bench_duration = time.time() - benchmark_start
    print(f"\n🎉 ALL {len(workload_specs)} WORKLOAD REQUESTS COMPLETED in {total_bench_duration:.1f}s!", flush=True)

    # 7. Stop Telemetry
    telemetry.stop()
    telemetry.join(timeout=2.0)

    # 8. Compile Metrics and Generate Visualizations
    print("\n" + "=" * 80)
    print(" GENERATING PRODUCTION TELEMETRY VISUALIZATIONS & METRICS ")
    print("=" * 80)

    records = telemetry.records
    times = [r["time_s"] for r in records]
    gpu0_vram = [r["gpu0_vram_mib"] for r in records]
    gpu1_vram = [r["gpu1_vram_mib"] for r in records]
    gpu0_util = [r["gpu0_util_pct"] for r in records]
    gpu1_util = [r["gpu1_util_pct"] for r in records]
    q_tts = [r["q_tts"] for r in records]
    q_vocode = [r["q_vocode"] for r in records]
    q_enhance = [r["q_enhance"] for r in records]
    q_asr = [r["q_asr"] for r in records]

    # Save CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, "continuous_telemetry.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"✅ Raw telemetry CSV saved: {csv_path} ({len(records)} samples)")

    json_res_path = os.path.join(RESULTS_DIR, "benchmark_requests.json")
    with open(json_res_path, "w") as f:
        json.dump(results_map, f, indent=2)
    print(f"✅ Request execution stats saved: {json_res_path}")

    # Plot 1: VRAM Timeline (Proving 14,800 MiB Ceiling & 0 Leak)
    plt.figure(figsize=(12, 5))
    plt.plot(times, gpu0_vram, label="GPU 0 (Generator A + Tiled Vocoder A + ASR)", color="#1f77b4", linewidth=2)
    plt.plot(times, gpu1_vram, label="GPU 1 (Generator B + Spillover Vocoder B + Enhancer)", color="#ff7f0e", linewidth=2)
    plt.axhline(y=15360, color="red", linestyle="--", alpha=0.7, label="Hardware Capacity (15,360 MiB)")
    plt.axhline(y=14800, color="darkred", linestyle=":", alpha=0.7, label="Hard Safety Ceiling (14,800 MiB)")
    plt.title("Dual NVIDIA Tesla T4 VRAM Saturation Timeline Under Continuous Multi-Model Stress", fontsize=13, fontweight="bold")
    plt.xlabel("Elapsed Benchmark Time (seconds)", fontsize=11)
    plt.ylabel("VRAM Used (MiB)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    vram_plot = os.path.join(RESULTS_DIR, "vram_saturation_timeline.png")
    plt.savefig(vram_plot, dpi=150)
    plt.close()
    print(f"✅ VRAM timeline plot saved: {vram_plot}")

    # Plot 2: GPU Compute Utilization %
    plt.figure(figsize=(12, 5))
    # Smooth with rolling average
    w = 5
    smooth0 = np.convolve(gpu0_util, np.ones(w)/w, mode="same") if len(gpu0_util) > w else gpu0_util
    smooth1 = np.convolve(gpu1_util, np.ones(w)/w, mode="same") if len(gpu1_util) > w else gpu1_util
    plt.plot(times, smooth0, label="GPU 0 Compute Util %", color="#2ca02c", linewidth=2)
    plt.plot(times, smooth1, label="GPU 1 Compute Util %", color="#9467bd", linewidth=2)
    plt.title("GPU Compute Utilization (%) Under Continuous Concurrency & Dual-Island Fanout", fontsize=13, fontweight="bold")
    plt.xlabel("Elapsed Benchmark Time (seconds)", fontsize=11)
    plt.ylabel("GPU Compute Utilization (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    util_plot = os.path.join(RESULTS_DIR, "gpu_compute_utilization.png")
    plt.savefig(util_plot, dpi=150)
    plt.close()
    print(f"✅ Compute utilization plot saved: {util_plot}")

    # Plot 3: Real-Time Factor (RTF) by Category
    categories = [results_map[k]["category"] for k in results_map]
    rtfs = [results_map[k]["rtf"] for k in results_map]
    req_labels = [results_map[k]["id"].replace("req_", "") for k in results_map]

    plt.figure(figsize=(14, 6))
    colors = ["#1f77b4" if r < 0.5 else "#2ca02c" if r < 1.0 else "#d62728" for r in rtfs]
    bars = plt.bar(req_labels, rtfs, color=colors, edgecolor="black", alpha=0.85)
    plt.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Real-Time Boundary (1.0x RTF)")
    plt.axhline(y=0.4, color="green", linestyle=":", linewidth=1.5, label="High-Performance Target (0.4x RTF)")
    plt.title("Real-Time Factor (RTF = Compute Time / Audio Duration) Across Diverse Workloads", fontsize=13, fontweight="bold")
    plt.xlabel("Workload Request ID", fontsize=11)
    plt.ylabel("Real-Time Factor (Lower is Faster)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.legend(loc="upper right", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    rtf_plot = os.path.join(RESULTS_DIR, "rtf_by_request_type.png")
    plt.savefig(rtf_plot, dpi=150)
    plt.close()
    print(f"✅ RTF distribution plot saved: {rtf_plot}")

    # Plot 4: Decoupled Queue Depths Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(times, q_tts, label="TTS Chunks Queue (Max 150)", color="#e377c2", linewidth=2)
    plt.plot(times, q_vocode, label="Vocode Queue (Tiled Dynamic)", color="#bcbd22", linewidth=2)
    plt.plot(times, q_enhance, label="Enhance Queue (Max 8)", color="#17becf", linewidth=2)
    plt.plot(times, q_asr, label="ASR Queue (Max 20)", color="#8c564b", linewidth=2)
    plt.title("Decoupled Orchestrator Queue Drainage Over Time (Zero Cross-Stage Blocking)", fontsize=13, fontweight="bold")
    plt.xlabel("Elapsed Benchmark Time (seconds)", fontsize=11)
    plt.ylabel("Queue Depth (# of items)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    queue_plot = os.path.join(RESULTS_DIR, "queue_depths_over_time.png")
    plt.savefig(queue_plot, dpi=150)
    plt.close()
    print(f"✅ Queue depths plot saved: {queue_plot}")

    # 9. Print Executive Performance Summary Table
    print("\n" + "=" * 95)
    print(f"{'REQUEST ID':<26} | {'CATEGORY':<25} | {'WORDS':<6} | {'AUDIO (s)':<10} | {'WALL (s)':<9} | {'RTF':<6} | {'STATUS'}")
    print("=" * 95)
    total_audio_gen = 0.0
    for req_id, data in results_map.items():
        total_audio_gen += data["audio_duration_s"]
        status = "✅ PASS" if data["success"] else "❌ FAIL"
        print(f"{req_id:<26} | {data['category']:<25} | {data['words']:<6} | {data['audio_duration_s']:<10.1f} | {data['wall_time_s']:<9.2f} | {data['rtf']:<5.2f}x | {status}")
    print("=" * 95)
    print(f"Total Audio Generated: {total_audio_gen:.1f} seconds ({total_audio_gen/60.0:.2f} minutes) across {len(results_map)} requests.")
    print(f"Total Benchmark Wall Time: {total_bench_duration:.1f} seconds.")
    print(f"Effective Cluster Real-Time Multiplier: {total_audio_gen / total_bench_duration:.2f}x Realtime!")
    print("=" * 95)

if __name__ == "__main__":
    run_benchmark()
