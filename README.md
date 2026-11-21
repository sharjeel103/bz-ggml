# `bz-ggml`: Distributed High-Throughput Speech Inference Engine

High-performance, decoupled, continuous-batching speech synthesis and audio processing engine for **Breeze-TTS-2**, powered by **GGML** on **Dual NVIDIA Tesla T4 GPUs**.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/hoppouai/breeze-tts-2-cpp-dual-t4)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: >=3.9](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![CUDA: sm_75](https://img.shields.io/badge/CUDA-Tesla%20T4%20(sm__75)-green.svg)]()

---

## 🚀 Key Highlights & Capabilities

- **Elastic Semantic Chunking**: Intelligently partitions long-form scripts at natural paragraph (`\n\n`) and sentence (`[.!?]`) boundaries with balanced lookahead instead of arbitrary word cuts. Scripts under 280 words execute as a single zero-overhead chunk.
- **Asynchronous Out-of-Order Assembly**: Chunks generated asynchronously across GPU 0 and GPU 1 are slotted into an indexed assembly manager and seamlessly stitched with a 200 ms breath pause and 5 ms micro-fades. Zero head-of-line blocking.
- **Pure Routing DAG Orchestrator**: Event-driven coordination using an internal completion queue ($< 10\mu\text{s}$ routing handoff). Supports arbitrary pipeline permutations (`TTS only`, `Enhance only`, `ASR only`, `Voice Registration`, or full multi-stage pipeline).
- **Decoupled Dedicated Stage Queues**: Independent worker pools calibrated to hardware limits with deterministic admission backpressure (`REJECTED_STAGE_OVERFLOW` with explicit wait ETA).
- **Dual Tesla T4 Continuous Batching**: Unlocked 2,040-step context and continuous iteration-level scheduling across GPU 0 and GPU 1.
- **Standalone 1.0s Telemetry Service**: Sub-microsecond (< 5 µs, < 0.05% CPU) telemetry daemon streaming real-time GPU VRAM, queue depths, and per-request progress percentages.
- **mTLS Control Plane Wire Contract (`workerpb`)**: Standardized Protobuf envelope implementation (`REGISTER`, `DISPATCH`, `ACK`, `STATUS`, `CANCEL`, `SYNC_REQUEST`) with zero audio streaming over WebSocket (direct presigned cloud bucket PUT with transactional fallback).

---

## 🏛️ System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │    Central Server / Control Plane       │
                  │   (mTLS WebSocket Client / Ingress)     │
                  └────────────────────┬────────────────────┘
                                       │ workerpb Envelopes / REST
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │          Admission Controller           │
                  │   Checks Stage Queues & Calibrated ETA  │
                  └────────────────────┬────────────────────┘
                                       │ Admitted Parcels
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │        Engine Orchestrator (DAG)        │
                  │     Non-Blocking Event Routing Loop     │
                  └────────┬───────────┬───────────┬────────┘
                           │           │           │
        ┌──────────────────┘           │           └──────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   tts_queue   │              │ enhance_queue │              │   asr_queue   │
│  (150 chunks) │              │   (8 jobs)    │              │   (20 jobs)   │
│ Continuous    │              │ Resemble 44.1k│              │ Whisper Turbo │
│ Batching G0/1 │              │ Conc=1 (GPU 1)│              │ Conc=1 (GPU 0)│
└───────┬───────┘              └───────┬───────┘              └───────┬───────┘
        │                              │                              │
        └──────────────────┬───────────┴───────────┬──────────────────┘
                           ▼                       ▼
                  ┌─────────────────┐    ┌──────────────────┐
                  │  master_queue   │    │   upload_queue   │
                  │    (20 jobs)    │    │   (100+ jobs)    │
                  │ FFmpeg Broadcast│    │ Presigned PUT    │
                  │  Conc=4 (CPU)   │    │ Conc=16 (Async)  │
                  └────────┬────────┘    └─────────┬────────┘
                           │                       │
                           └───────────┬───────────┘
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │       Completion Queue / Inbox          │
                  │      Emits Final JobDoneEvent           │
                  └─────────────────────────────────────────┘
```

---

## 📊 Hardware Queue Ceilings & Concurrency

| Stage Queue | Ceiling | Concurrency | Hardware Target | Responsibilities / Processing Specs |
| :--- | :---: | :---: | :--- | :--- |
| **`tts_queue`** | **150 Chunks** | Dynamic | GPU 0 & GPU 1 | Continuous batching across dual T4 instances. Token budget up to 50k tokens. |
| **`enhance_queue`** | **8 Jobs** | **1** | GPU 1 (CUDA 1) | Resemble 44.1 kHz enhancement. 10s sliding windows (~6.5s per job $\to$ max 52s wait). |
| **`asr_queue`** | **20 Jobs** | **1** | GPU 0 (CUDA 0) | Faster-Whisper Turbo (~0.2s per job). Concurrency = 1 protects GPU 0 VRAM during active TTS. |
| **`voice_reg_queue`**| **15 Jobs** | **1** | GPU 0 (CUDA 0) | Breeze acoustic reference encoder pool (~1.2s per 10s clip). |
| **`master_queue`** | **20 Jobs** | **4** | CPU Cores | FFmpeg broadcast filter chain, loudnorm, studio mastering (~0.5s per job). |
| **`upload_queue`** | **100+ Jobs** | **16** | Async Network | Direct presigned HTTP PUT uploads to cloud buckets (0 MB GPU memory). |

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/sharjeel103/bz-ggml.git
cd bz-ggml

# Install core dependencies
pip install -r requirements.txt

# Or install editable package with GPU & UI support
pip install -e ".[all]"
```

### 2. Turnkey Dual Tesla T4 Setup (< 15 Seconds)
Precompiled CUDA `sm_75` binaries are provided for instant Kaggle startup without compiling:
```bash
# Download precompiled binaries (~39 MB)
wget -q https://github.com/sharjeel103/bz-ggml/raw/prebuilt-t4/breeze-cuda-sm75.tar.gz
tar -xzf breeze-cuda-sm75.tar.gz
mkdir -p build && cp -P breeze-cuda-sm75/lib/* build/
```

### 3. Headless Production Pipeline Example
```python
from bz_ggml import DualInstanceCluster, EngineOrchestrator, Parcel, TaskType, TaskSpec

# Initialize Dual-GPU Cluster across GPU 0 & GPU 1
cluster = DualInstanceCluster(
    model_path="models/breeze-tts-2-f16.gguf",
    lib_path="build/libbreeze.so",
    enable_vocoder=True
)

# Initialize Pure Decoupled Orchestrator
orchestrator = EngineOrchestrator(cluster=cluster)

# Build a Multi-Stage Pipeline Parcel
parcel = Parcel(
    parcel_id="req_001",
    tasks=[
        TaskSpec(TaskType.TTS),
        TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}),
        TaskSpec(TaskType.ASR, {"output_format": "txt"}),
        TaskSpec(TaskType.MASTER)
    ],
    payload={
        "script_text": "The distributed inference engine is fully operational with decoupled stage queues.",
        "instruction": "Speak with natural clarity and professional authority."
    },
    return_artifacts=["audio_wav", "audio_mp3", "subtitles_srt", "transcript_txt"]
)

# Dispatch asynchronously or synchronously
result = orchestrator.dispatch(parcel)
print(f"Generated {result.duration_s}s audio in {result.stage_timings}")
print(f"Verbatim ASR transcript: {result.transcript}")
```

### 4. Interactive Gradio Studio
```bash
python3 python/gradio_app.py \
    --model models/breeze-tts-2-f16.gguf \
    --lib build/libbreeze.so \
    --vocoder-mode on \
    --asr-device 0 \
    --share --port 7860
```

---

## 🧪 Testing & Verification

Run the full automated test suite:
```bash
# Comprehensive 8-point architecture test suite
PYTHONPATH=python python3 tests/test_suite_comprehensive.py

# Voice persona serialization roundtrip test
PYTHONPATH=python python3 tests/test_voice_serialization.py

# ASR memory footprint test (on CUDA host)
PYTHONPATH=python python3 tests/test_asr_memory_footprint.py
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
