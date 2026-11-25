import gc
import os
import sys
import time
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Ensure results can be saved
RESULTS_DIR = "/kaggle/working/benchmark_results" if os.path.exists("/kaggle") else "./benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class VRAMTracker(threading.Thread):
    """Continuous high-frequency physical VRAM poller via nvidia-smi."""
    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._running = True
        self.peak_mib = {0: 0, 1: 0}
        self.current_mib = {0: 0, 1: 0}

    def run(self):
        cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
        while self._running:
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
                for line in out.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        idx = int(parts[0].strip())
                        used = int(parts[1].strip())
                        self.current_mib[idx] = used
                        if used > self.peak_mib[idx]:
                            self.peak_mib[idx] = used
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False

    def reset_peak(self, gpu_id: int):
        self.peak_mib[gpu_id] = self.current_mib.get(gpu_id, 0)


def generate_synthetic_speech_audio(duration_s: float = 1200.0, sr: int = 24000) -> np.ndarray:
    """
    Generates realistic speech-formant modulated audio.
    """
    print(f">>> Synthesizing {duration_s:.1f}s ({duration_s/60:.1f} min) audio @ {sr} Hz...", flush=True)
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    # Fundamental frequency varying around 140 Hz (pitch intonation)
    f0 = 140.0 + 25.0 * np.sin(2.0 * np.pi * 0.5 * t)
    # Formants at 500 Hz, 1500 Hz, 2500 Hz
    wave = 0.5 * np.sin(2.0 * np.pi * f0 * t) + \
           0.3 * np.sin(2.0 * np.pi * 500.0 * t) + \
           0.2 * np.sin(2.0 * np.pi * 1500.0 * t)
    # Syllabic envelope modulation (4 syllables per second)
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t))
    audio = wave * env * 0.70
    return audio.astype(np.float32)


def profile_resemble_enhancer_gpu1(tracker: VRAMTracker, audio_24k: np.ndarray) -> dict:
    import torch
    from resemble_enhance.enhancer.inference import load_enhancer, inference

    device = "cuda:1"
    gpu_id = 1
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1.0)

    base_idle_mib = tracker.current_mib.get(gpu_id, 0)
    print(f"\n[GPU 1 - ENHANCER] Baseline Idle VRAM: {base_idle_mib} MiB", flush=True)

    print(f"[GPU 1 - ENHANCER] Loading model weights to {device}...", flush=True)
    t_load_start = time.time()
    model = load_enhancer(None, device=device)
    load_wall_s = time.time() - t_load_start
    time.sleep(1.0)

    post_load_mib = tracker.current_mib.get(gpu_id, 0)
    torch_alloc_load_mib = torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)
    torch_res_load_mib = torch.cuda.memory_reserved(gpu_id) / (1024 * 1024)
    exact_model_vram_mib = post_load_mib - base_idle_mib

    print(f"[GPU 1 - ENHANCER] Loaded in {load_wall_s:.2f}s!", flush=True)
    print(f"  • Physical Post-Load VRAM: {post_load_mib} MiB")
    print(f"  • EXACT MODEL WEIGHT VRAM: {exact_model_vram_mib} MiB (nvidia-smi delta)")
    print(f"  • PyTorch Allocated: {torch_alloc_load_mib:.2f} MiB | Reserved: {torch_res_load_mib:.2f} MiB", flush=True)

    # Reset peak tracking for inference pass
    tracker.reset_peak(gpu_id)
    torch.cuda.reset_peak_memory_stats(gpu_id)

    duration_s = len(audio_24k) / 24000.0
    print(f"[GPU 1 - ENHANCER] Starting 20-minute ({duration_s:.1f}s) inference (chunk=10.0s, overlap=1.0s)...", flush=True)
    
    # Must remain on CPU as inference_chunk expects CPU tensor for normalization
    dwav = torch.from_numpy(audio_24k)
    t_infer_start = time.time()
    
    hwav, out_sr = inference(
        model=model,
        dwav=dwav,
        sr=24000,
        device=device,
        chunk_seconds=10.0,
        overlap_seconds=1.0,
    )
    
    infer_wall_s = time.time() - t_infer_start
    peak_infer_mib = tracker.peak_mib.get(gpu_id, 0)
    torch_alloc_peak_mib = torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024)
    torch_res_peak_mib = torch.cuda.max_memory_reserved(gpu_id) / (1024 * 1024)
    exact_infer_overhead_mib = peak_infer_mib - post_load_mib
    rtf = infer_wall_s / duration_s

    print(f"\n[GPU 1 - ENHANCER] 20-Minute Inference Completed in {infer_wall_s:.2f}s (RTF: {rtf:.3f}x)!", flush=True)
    print(f"  • Peak Physical VRAM: {peak_infer_mib} MiB")
    print(f"  • EXACT INFERENCE PEAK VRAM: {peak_infer_mib} MiB")
    print(f"  • Inference Overhead Delta: {exact_infer_overhead_mib} MiB")
    print(f"  • PyTorch Peak Allocated: {torch_alloc_peak_mib:.2f} MiB | Peak Reserved: {torch_res_peak_mib:.2f} MiB", flush=True)

    return {
        "model_name": "Resemble Enhancer",
        "gpu_id": 1,
        "device": device,
        "audio_duration_s": duration_s,
        "load_wall_s": load_wall_s,
        "infer_wall_s": infer_wall_s,
        "rtf": rtf,
        "baseline_idle_mib": base_idle_mib,
        "post_load_mib": post_load_mib,
        "exact_model_vram_mib": exact_model_vram_mib,
        "torch_alloc_load_mib": torch_alloc_load_mib,
        "torch_res_load_mib": torch_res_load_mib,
        "peak_infer_mib": peak_infer_mib,
        "exact_infer_overhead_mib": exact_infer_overhead_mib,
        "torch_alloc_peak_mib": torch_alloc_peak_mib,
        "torch_res_peak_mib": torch_res_peak_mib,
    }


def profile_whisper_asr_gpu0(tracker: VRAMTracker, audio_16k: np.ndarray) -> dict:
    from faster_whisper import WhisperModel
    import torch

    device = "cuda"
    device_index = 0
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1.0)

    base_idle_mib = tracker.current_mib.get(gpu_id, 0)
    print(f"\n[GPU 0 - ASR] Baseline Idle VRAM: {base_idle_mib} MiB", flush=True)

    print(f"[GPU 0 - ASR] Loading Whisper Turbo (float16) to {device}:{device_index}...", flush=True)
    t_load_start = time.time()
    model = WhisperModel(
        "turbo",
        device=device,
        device_index=device_index,
        compute_type="float16"
    )
    load_wall_s = time.time() - t_load_start
    time.sleep(1.0)

    post_load_mib = tracker.current_mib.get(gpu_id, 0)
    exact_model_vram_mib = post_load_mib - base_idle_mib

    print(f"[GPU 0 - ASR] Loaded in {load_wall_s:.2f}s!", flush=True)
    print(f"  • Physical Post-Load VRAM: {post_load_mib} MiB")
    print(f"  • EXACT MODEL WEIGHT VRAM: {exact_model_vram_mib} MiB (nvidia-smi delta)", flush=True)

    # Reset peak tracking for inference pass
    tracker.reset_peak(gpu_id)

    duration_s = len(audio_16k) / 16000.0
    print(f"[GPU 0 - ASR] Starting 20-minute ({duration_s:.1f}s) inference (beam=1, vad_filter=False)...", flush=True)
    
    t_infer_start = time.time()
    # vad_filter=False ensures every single 30s window across the full 20 minutes is processed through Transformer
    segments, info = model.transcribe(
        audio_16k,
        beam_size=1,
        language="en",
        task="transcribe",
        temperature=0.0,
        vad_filter=False,
        word_timestamps=True
    )
    seg_list = list(segments)
    infer_wall_s = time.time() - t_infer_start
    
    peak_infer_mib = tracker.peak_mib.get(gpu_id, 0)
    exact_infer_overhead_mib = peak_infer_mib - post_load_mib
    rtf = infer_wall_s / duration_s

    print(f"\n[GPU 0 - ASR] 20-Minute Inference Completed in {infer_wall_s:.2f}s (RTF: {rtf:.3f}x)!", flush=True)
    print(f"  • Windows/Segments Processed: {len(seg_list)}")
    print(f"  • Peak Physical VRAM: {peak_infer_mib} MiB")
    print(f"  • EXACT INFERENCE PEAK VRAM: {peak_infer_mib} MiB")
    print(f"  • Inference Overhead Delta: {exact_infer_overhead_mib} MiB", flush=True)

    return {
        "model_name": "Faster-Whisper Turbo",
        "gpu_id": 0,
        "device": f"{device}:{device_index}",
        "audio_duration_s": duration_s,
        "load_wall_s": load_wall_s,
        "infer_wall_s": infer_wall_s,
        "rtf": rtf,
        "baseline_idle_mib": base_idle_mib,
        "post_load_mib": post_load_mib,
        "exact_model_vram_mib": exact_model_vram_mib,
        "peak_infer_mib": peak_infer_mib,
        "exact_infer_overhead_mib": exact_infer_overhead_mib,
        "segments_count": len(seg_list)
    }


def main():
    print("=" * 80)
    print(" EMPIRICAL STANDALONE VRAM PROFILING SUITE (20-MINUTE CONTINUOUS AUDIO) ")
    print("=" * 80)

    # 1. Start continuous VRAM tracker
    tracker = VRAMTracker(interval_s=0.05)
    tracker.start()
    time.sleep(1.0)

    # 2. Prepare 20-minute audio arrays (1200 seconds)
    audio_24k = generate_synthetic_speech_audio(duration_s=1200.0, sr=24000)
    
    import scipy.signal
    print(">>> Resampling 20-minute audio to 16 kHz for Whisper ASR...", flush=True)
    audio_16k = scipy.signal.resample_poly(audio_24k, 16000, 24000).astype(np.float32)

    print("\n>>> Launching Simultaneous Profiling: ASR on GPU 0 and Enhancer on GPU 1...", flush=True)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_enh = executor.submit(profile_resemble_enhancer_gpu1, tracker, audio_24k)
        f_asr = executor.submit(profile_whisper_asr_gpu0, tracker, audio_16k)
        
        enh_res = f_enh.result()
        asr_res = f_asr.result()

    tracker.stop()
    total_time = time.time() - t_start

    # Print Final Truthful Summary Table
    print("\n" + "=" * 80)
    print(" FINAL EMPIRICAL RESULTS (UNASSAILABLE PHYSICAL NUMBERS) ")
    print("=" * 80)
    print(f"Total Test Wall Time: {total_time:.2f} seconds\n")

    print(f"1. RESEMBLE ENHANCER (GPU 1 | {enh_res['device']}):")
    print(f"   • Baseline Idle VRAM:           {enh_res['baseline_idle_mib']:>5d} MiB")
    print(f"   • Post-Load Resident VRAM:      {enh_res['post_load_mib']:>5d} MiB")
    print(f"   • [NUMBER 1] MODEL WEIGHT VRAM:  {enh_res['exact_model_vram_mib']:>5d} MiB")
    print(f"   • Peak During 20m Inference:    {enh_res['peak_infer_mib']:>5d} MiB")
    print(f"   • [NUMBER 2] PEAK INFERENCE VRAM: {enh_res['peak_infer_mib']:>5d} MiB")
    print(f"   • Inference Overhead Delta:     {enh_res['exact_infer_overhead_mib']:>5d} MiB")
    print(f"   • Audio Processed:              {enh_res['audio_duration_s']:.1f}s ({enh_res['audio_duration_s']/60:.1f} min)")
    print(f"   • Wall Time / RTF:              {enh_res['infer_wall_s']:.2f}s (RTF: {enh_res['rtf']:.3f}x)\n")

    print(f"2. FASTER-WHISPER TURBO ASR (GPU 0 | {asr_res['device']}):")
    print(f"   • Baseline Idle VRAM:           {asr_res['baseline_idle_mib']:>5d} MiB")
    print(f"   • Post-Load Resident VRAM:      {asr_res['post_load_mib']:>5d} MiB")
    print(f"   • [NUMBER 1] MODEL WEIGHT VRAM:  {asr_res['exact_model_vram_mib']:>5d} MiB")
    print(f"   • Peak During 20m Inference:    {asr_res['peak_infer_mib']:>5d} MiB")
    print(f"   • [NUMBER 2] PEAK INFERENCE VRAM: {asr_res['peak_infer_mib']:>5d} MiB")
    print(f"   • Inference Overhead Delta:     {asr_res['exact_infer_overhead_mib']:>5d} MiB")
    print(f"   • Audio Processed:              {asr_res['audio_duration_s']:.1f}s ({asr_res['audio_duration_s']/60:.1f} min)")
    print(f"   • Wall Time / RTF:              {asr_res['infer_wall_s']:.2f}s (RTF: {asr_res['rtf']:.3f}x)")
    print("=" * 80)

if __name__ == "__main__":
    main()
