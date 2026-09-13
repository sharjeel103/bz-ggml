import os
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from .bindings import BreezeLib, GeneratorHandle, VocoderHandle

@dataclass
class UserTask:
    id: int
    text: str
    instruction: str = "Speak clearly and naturally."
    seed: int = 42

@dataclass
class ClusterResult:
    id: int
    words: int
    audio_s: float
    queue_wait_s: float
    compute_wall_s: float
    turnaround_s: float
    ttfa_s: float
    rtf: float
    worker: str
    wav_path: str

class DualInstanceCluster:
    """
    Dual-Instance Generator & Dedicated Streaming Vocoder Cluster.
      GPU 0: Text Encoder + Generator Instance A (Backbone + 15-step Depth)
      GPU 1: Generator Instance B (Backbone + 15-step Depth) + Dedicated Streaming Vocoder
      Zero Intra-Frame PCIe Ping-Pong | Sub-250ms Warm TTFA | 8-Frame Batched Chunks
    """

    def __init__(self, model_path: str, lib_path: Optional[str] = None):
        self.model_path = model_path
        self.lib = BreezeLib(lib_path)

        print("[Cluster] Initializing Generator Instance A on CUDA0...")
        t0 = time.time()
        self.gen_a = GeneratorHandle(self.lib, model_path, cuda_device=0)
        print(f"   -> Instance A Online on CUDA0 in {time.time()-t0:.2f}s")

        print("[Cluster] Initializing Generator Instance B on CUDA1...")
        t0 = time.time()
        self.gen_b = GeneratorHandle(self.lib, model_path, cuda_device=1)
        print(f"   -> Instance B Online on CUDA1 in {time.time()-t0:.2f}s")

        print("[Cluster] Initializing Dedicated Streaming Vocoder on CUDA1...")
        t0 = time.time()
        self.voc = VocoderHandle(self.lib, model_path, cuda_device=1)
        print(f"   -> Streaming Vocoder Online on CUDA1 in {time.time()-t0:.2f}s")

    def run_workload(
        self,
        tasks: List[UserTask],
        out_dir: str = "audio_out",
        arrival_delays: Optional[List[float]] = None,
        on_progress: Optional[Callable[[ClusterResult], None]] = None
    ) -> List[ClusterResult]:
        os.makedirs(out_dir, exist_ok=True)

        job_queue = queue.Queue()
        vocoder_queue = queue.Queue()
        user_audio_results = {}
        user_audio_lock = threading.Lock()
        completed_results: List[ClusterResult] = []
        results_lock = threading.Lock()
        cluster_t0 = time.time()
        workers_stopping = False

        # 1. Vocoder Worker Thread on GPU 1
        def vocoder_loop():
            while not workers_stopping:
                try:
                    task = vocoder_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if task is None:
                    break

                u_id, words, arr_time, t_start, frames_list, is_first, is_eos, total_frames, worker_tag = task

                n_frames = len(frames_list) // 16
                if n_frames > 0:
                    samples = self.voc.stream_decode(frames_list, n_frames)
                    with user_audio_lock:
                        if u_id not in user_audio_results:
                            user_audio_results[u_id] = {
                                "samples": [],
                                "first_audio_time": None
                            }
                        user_audio_results[u_id]["samples"].extend(samples)
                        if is_first and user_audio_results[u_id]["first_audio_time"] is None:
                            user_audio_results[u_id]["first_audio_time"] = time.time()

                if is_eos:
                    t_end = time.time()
                    with user_audio_lock:
                        u_rec = user_audio_results.get(u_id, {})
                        pcm_samples = u_rec.get("samples", [])
                        fa_time = u_rec.get("first_audio_time", t_end)

                    audio_s = len(pcm_samples) / 24000.0
                    wall_s = t_end - t_start
                    turnaround_s = t_end - arr_time
                    wait_s = t_start - arr_time
                    ttfa = (fa_time - arr_time) if fa_time else turnaround_s
                    rtf = wall_s / (audio_s if audio_s > 0 else 1.0)

                    # Save WAV
                    out_wav = os.path.join(out_dir, f"dual_gen_user_{u_id:02d}.wav")
                    with wave.open(out_wav, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(24000)
                        pcm_np = np.array(pcm_samples, dtype=np.float32)
                        pcm_int16 = (np.clip(pcm_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                        w.writeframes(pcm_int16.tobytes())

                    res = ClusterResult(
                        id=u_id,
                        words=words,
                        audio_s=round(audio_s, 2),
                        queue_wait_s=round(wait_s, 2),
                        compute_wall_s=round(wall_s, 2),
                        turnaround_s=round(turnaround_s, 2),
                        ttfa_s=round(ttfa, 3),
                        rtf=round(rtf, 3),
                        worker=worker_tag,
                        wav_path=out_wav
                    )

                    with results_lock:
                        completed_results.append(res)
                        if on_progress:
                            on_progress(res)

                vocoder_queue.task_done()

        voc_thread = threading.Thread(target=vocoder_loop)
        voc_thread.start()

        # 2. Generator Worker Loop
        def generator_loop(worker_name: str, gen: GeneratorHandle, gpu_id: int):
            while not workers_stopping:
                try:
                    task_item, arr_time = job_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if task_item is None:
                    break

                t_exec_start = time.time()
                words = len(task_item.text.split())

                # Stage 1: Prefill
                cb0 = gen.prefill(task_item.text, task_item.instruction, task_item.seed + task_item.id)

                # Stage 2: Frame Loop
                chunk_buffer = []
                chunk_size = 8
                total_frames = 0
                max_steps = 700

                for step in range(max_steps):
                    if cb0 < 0:
                        break

                    next_cb0, frame16 = gen.step_frame(cb0, task_item.seed + step)
                    total_frames += 1
                    chunk_buffer.extend(frame16)

                    # Immediate Frame 1 dispatch for low TTFA
                    if total_frames == 1:
                        vocoder_queue.put((
                            task_item.id, words, arr_time, t_exec_start,
                            list(chunk_buffer), True, False, total_frames, f"GPU {gpu_id} ({worker_name})"
                        ))
                        chunk_buffer = []
                    elif len(chunk_buffer) >= (chunk_size * 16):
                        vocoder_queue.put((
                            task_item.id, words, arr_time, t_exec_start,
                            list(chunk_buffer), False, False, total_frames, f"GPU {gpu_id} ({worker_name})"
                        ))
                        chunk_buffer = []

                    cb0 = next_cb0

                # End of stream flush
                vocoder_queue.put((
                    task_item.id, words, arr_time, t_exec_start,
                    list(chunk_buffer), False, True, total_frames, f"GPU {gpu_id} ({worker_name})"
                ))
                job_queue.task_done()

        worker_a = threading.Thread(target=generator_loop, args=("Instance A", self.gen_a, 0))
        worker_b = threading.Thread(target=generator_loop, args=("Instance B", self.gen_b, 1))
        worker_a.start()
        worker_b.start()

        # 3. Dispatcher
        t_dispatch0 = time.time()
        for idx, task in enumerate(tasks):
            delay = arrival_delays[idx] if arrival_delays and idx < len(arrival_delays) else 0.0
            if delay > 0:
                elapsed = time.time() - t_dispatch0
                if delay > elapsed:
                    time.sleep(delay - elapsed)
            job_queue.put((task, time.time()))

        job_queue.join()
        vocoder_queue.join()

        # Shutdown workers
        workers_stopping = True
        job_queue.put((None, None))
        job_queue.put((None, None))
        vocoder_queue.put(None)
        worker_a.join()
        worker_b.join()
        voc_thread.join()

        completed_results.sort(key=lambda x: x.id)
        return completed_results

    def close(self):
        if hasattr(self, "gen_a") and self.gen_a:
            self.gen_a.close()
        if hasattr(self, "gen_b") and self.gen_b:
            self.gen_b.close()
        if hasattr(self, "voc") and self.voc:
            self.voc.close()

    def __del__(self):
        self.close()
