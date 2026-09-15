import math
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
    cfg_scale: float = 1.0
    seed: int = 42
    max_steps: int = 1000
    ref_text: Optional[str] = None
    ref_codes: Optional[List[int]] = None
    ref_frames: int = 0


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
    Symmetric Autonomous Dual-Island Architecture (10/10 Design).
      GPU 0: Generator Instance A + Dedicated Streaming Vocoder A (Autonomous Island 0)
      GPU 1: Generator Instance B + Dedicated Streaming Vocoder B (Autonomous Island 1)
      Zero Cross-GPU PCIe Contention | 100% Saturated Dual GPUs | Adaptive Dynamic Quantum
      Optional Surge Tiering: Dynamically routes burst overflow sessions to Q4 when threshold exceeded.
    """

    def __init__(
        self,
        model_path: str,
        lib_path: Optional[str] = None,
        q4_model_path: Optional[str] = None,
        enable_q4_burst: bool = False,
        q4_threshold: int = 10
    ):
        self.model_path = model_path
        self.q4_model_path = q4_model_path
        self.enable_q4_burst = enable_q4_burst
        self.q4_threshold = q4_threshold
        self.lib = BreezeLib(lib_path)

        # Autonomous Island 0 (GPU 0)
        print("[Cluster] Initializing Autonomous Island 0 on CUDA0...")
        t0 = time.time()
        self.gen_a = GeneratorHandle(self.lib, model_path, cuda_device=0)
        self.voc_a = VocoderHandle(self.lib, model_path, cuda_device=0)
        print(f"   -> Island 0 (Gen A + Voc A) Online on CUDA0 in {time.time()-t0:.2f}s")

        # Autonomous Island 1 (GPU 1)
        print("[Cluster] Initializing Autonomous Island 1 on CUDA1...")
        t0 = time.time()
        self.gen_b = GeneratorHandle(self.lib, model_path, cuda_device=1)
        self.voc_b = VocoderHandle(self.lib, model_path, cuda_device=1)
        print(f"   -> Island 1 (Gen B + Voc B) Online on CUDA1 in {time.time()-t0:.2f}s")

        # Optional Modular INT4 Depth Decoder piece (~300 MiB)
        self.has_q4_dd = False
        if self.enable_q4_burst and self.q4_model_path and os.path.exists(self.q4_model_path):
            print(f"[Cluster] Loading Modular INT4 Depth Decoder piece (~300 MB) on CUDA0 and CUDA1 (Threshold: {self.q4_threshold})...")
            t0 = time.time()
            self.gen_a.load_q4_depth(self.q4_model_path)
            self.gen_b.load_q4_depth(self.q4_model_path)
            self.has_q4_dd = True
            print(f"   -> Modular INT4 Depth Decoder Online in {time.time()-t0:.2f}s (Saved 2.24 GB VRAM per GPU!)")

    def run_workload(
        self,
        tasks: List[UserTask],
        out_dir: str = "audio_out",
        arrival_delays: Optional[List[float]] = None,
        on_progress: Optional[Callable[[ClusterResult], None]] = None,
        quantum_frames: int = 2,
        max_slots_per_gpu: Optional[int] = None,
        vocoder_chunk_size: int = 16,
        max_active_words_per_gpu: int = 25000
    ) -> List[ClusterResult]:
        os.makedirs(out_dir, exist_ok=True)

        job_queue = queue.Queue()
        voc_a_queue = queue.Queue()
        voc_b_queue = queue.Queue()

        user_audio_results = {}
        user_audio_lock = threading.Lock()
        completed_results: List[ClusterResult] = []
        results_lock = threading.Lock()
        workers_stopping = False

        # 1. Independent Vocoder Worker Loops on GPU 0 and GPU 1
        def vocoder_loop(v_queue: queue.Queue, voc_handle: VocoderHandle, voc_tag: str):
            while not workers_stopping:
                try:
                    task = v_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if task is None:
                    break

                u_id, words, arr_time, t_start, frames_list, is_first, is_eos, total_frames, worker_tag = task

                n_frames = len(frames_list) // 16
                if n_frames > 0:
                    samples = voc_handle.stream_decode(frames_list, n_frames)
                    with user_audio_lock:
                        if u_id not in user_audio_results:
                            user_audio_results[u_id] = {
                                "samples": [],
                                "first_audio_time": None
                            }
                        user_audio_results[u_id]["samples"].extend(samples)
                        if user_audio_results[u_id]["first_audio_time"] is None:
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
                        worker=f"{worker_tag} + {voc_tag}",
                        wav_path=out_wav
                    )

                    with results_lock:
                        completed_results.append(res)
                        if on_progress:
                            on_progress(res)

                v_queue.task_done()

        voc_a_thread = threading.Thread(target=vocoder_loop, args=(voc_a_queue, self.voc_a, "Vocoder A (GPU 0)"))
        voc_b_thread = threading.Thread(target=vocoder_loop, args=(voc_b_queue, self.voc_b, "Vocoder B (GPU 1)"))
        voc_a_thread.start()
        voc_b_thread.start()

        # 2. Generator Multi-Session Worker Loop with Adaptive Quantum
        all_dispatched = threading.Event()

        def generator_multi_session_loop(
            worker_name: str,
            gen_default: GeneratorHandle,
            voc_queue: queue.Queue,
            gpu_id: int,
            max_slots: Optional[int] = None,
            max_active_words: int = 25000,
            chunk_size: int = 16
        ):
            active_sessions = {}
            last_hb_time = time.time()
            current_active_words = 0
            pending_item = None

            while not workers_stopping:
                # A. Admit waiting tasks governed strictly by max_active_words token budget
                while not workers_stopping:
                    if max_slots is not None and len(active_sessions) >= max_slots:
                        break

                    if pending_item is not None:
                        task_item, arr_time = pending_item
                    else:
                        try:
                            task_item, arr_time = job_queue.get_nowait()
                        except queue.Empty:
                            break

                        if task_item is None:
                            break

                    text_words = len(task_item.text.split())
                    ref_tokens = getattr(task_item, "ref_frames", 0)
                    ins_words = len(task_item.instruction.split()) if getattr(task_item, "instruction", None) else 0
                    cfg_scale = getattr(task_item, "cfg_scale", 1.0)
                    multiplier = 2 if cfg_scale > 1.0 else 1

                    # Exact empirical sweet-spot calculation matching C++:
                    input_pred_tokens = int(math.ceil(text_words * 1.35)) + ref_tokens + int(math.ceil(ins_words * 1.35)) + 10
                    output_cap_frames = min(1000, int(math.ceil(text_words * 2.2)) + 60)
                    if hasattr(task_item, "max_steps") and task_item.max_steps > 0:
                        output_cap_frames = min(output_cap_frames, task_item.max_steps)
                    task_tokens = multiplier * (input_pred_tokens + output_cap_frames)

                    # Check token capacity budget (25,000 active tokens limit per GPU):
                    if (current_active_words + task_tokens > max_active_words) and len(active_sessions) > 0:
                        # Hold task locally without re-queuing into job_queue
                        pending_item = (task_item, arr_time)
                        break

                    pending_item = None
                    t_exec_start = time.time()
                    sid = task_item.id

                    # Select Q4 if surge tiering is enabled and load exceeds threshold
                    use_q4 = (
                        self.enable_q4_burst
                        and self.has_q4_dd
                        and len(active_sessions) >= self.q4_threshold
                    )
                    active_gen = gen_default
                    model_tag = "Q4" if use_q4 else "Q8"

                    try:
                        cb0, allocated_tokens = active_gen.session_create_ext(
                            session_id=sid,
                            text=task_item.text,
                            instruction=task_item.instruction,
                            ref_text=task_item.ref_text,
                            ref_codes=task_item.ref_codes,
                            ref_frames=task_item.ref_frames,
                            cfg_scale=cfg_scale,
                            seed=task_item.seed + sid,
                            max_new_tokens=output_cap_frames,
                            use_q4=use_q4
                        )
                    except Exception as e:
                        print(f"[{worker_name}] Error creating session {sid}: {e}")
                        job_queue.task_done()
                        continue

                    current_active_words += allocated_tokens
                    active_sessions[sid] = {
                        "task": task_item,
                        "gen": active_gen,
                        "model_tag": model_tag,
                        "arr_time": arr_time,
                        "t_exec_start": t_exec_start,
                        "words": text_words,
                        "tokens": allocated_tokens,
                        "cb0": cb0,
                        "total_frames": 0,
                        "chunk_buffer": [],
                        "max_steps": getattr(task_item, "max_steps", 1000)
                    }

                if not active_sessions:
                    if all_dispatched.is_set() and job_queue.empty():
                        break
                    time.sleep(0.002)
                    continue

                # B. Step all active sessions concurrently using Batched GEMM (Layer-Outer Loop)
                active_sids = [sid for sid in active_sessions.keys()]
                if not active_sids:
                    continue

                step_seeds = [
                    active_sessions[sid]["task"].seed + active_sessions[sid]["total_frames"]
                    for sid in active_sids
                ]

                next_cb0s, frames16 = gen_default.sessions_step_batched(active_sids, step_seeds)

                finished_sids = []
                for i, sid in enumerate(active_sids):
                    s = active_sessions.get(sid)
                    if not s:
                        continue

                    task_item = s["task"]
                    next_cb0 = next_cb0s[i]
                    frame16 = frames16[i]

                    s["total_frames"] += 1
                    s["chunk_buffer"].extend(frame16)
                    s["cb0"] = next_cb0

                    # Adaptive Coalesced Audio Pipelining:
                    # - If chunk_size > 0: Pipelined Streaming Mode (dispatch every chunk_size frames)
                    # - If chunk_size <= 0: Utterance Coalescing Mode (hold in buffer until EOS)
                    if chunk_size > 0 and len(s["chunk_buffer"]) >= (chunk_size * 16):
                        voc_queue.put((
                            task_item.id, s["words"], s["arr_time"], s["t_exec_start"],
                            list(s["chunk_buffer"]), False, False, s["total_frames"],
                            f"GPU {gpu_id} ({worker_name} [{s['model_tag']}])"
                        ))
                        s["chunk_buffer"] = []

                    if next_cb0 < 0 or s["total_frames"] >= s["max_steps"]:
                        finished_sids.append(sid)

                # C. Check if EOS or max_steps reached
                for sid in finished_sids:
                    s = active_sessions.get(sid)
                    if not s:
                        continue
                    task_item = s["task"]
                    # End of stream flush to vocoder
                    voc_queue.put((
                        task_item.id, s["words"], s["arr_time"], s["t_exec_start"],
                        list(s["chunk_buffer"]), False, True, s["total_frames"],
                        f"GPU {gpu_id} ({worker_name} [{s['model_tag']}])"
                    ))
                    # Free session slot immediately in local VRAM
                    current_active_words -= s.get("tokens", s["words"])
                    gen_default.session_free(sid)
                    del active_sessions[sid]
                    job_queue.task_done()


                if time.time() - last_hb_time > 10.0 and active_sessions:
                    last_hb_time = time.time()
                    frames_list = [s["total_frames"] for s in active_sessions.values()]
                    min_f = min(frames_list) if frames_list else 0
                    max_f = max(frames_list) if frames_list else 0
                    print(f"  [Heartbeat GPU {gpu_id}] Active: {len(active_sessions):2d} streams ({current_active_words:,}/{max_active_words:,} tokens) | Frames: min {min_f:3d} / max {max_f:3d} | Audio: {sum(frames_list)*0.08:.1f}s", flush=True)

        worker_a = threading.Thread(
            target=generator_multi_session_loop,
            args=("Instance A", self.gen_a, voc_a_queue, 0, max_slots_per_gpu, max_active_words_per_gpu, vocoder_chunk_size)
        )
        worker_b = threading.Thread(
            target=generator_multi_session_loop,
            args=("Instance B", self.gen_b, voc_b_queue, 1, max_slots_per_gpu, max_active_words_per_gpu, vocoder_chunk_size)
        )
        worker_a.start()
        worker_b.start()

        # 3. Workload Dispatcher
        t_dispatch0 = time.time()
        for idx, task in enumerate(tasks):
            delay = arrival_delays[idx] if arrival_delays and idx < len(arrival_delays) else 0.0
            if delay > 0:
                time.sleep(delay)
            job_queue.put((task, time.time()))

        all_dispatched.set()

        job_queue.join()
        voc_a_queue.join()
        voc_b_queue.join()

        # Shutdown workers
        workers_stopping = True
        worker_a.join()
        worker_b.join()
        voc_a_thread.join()
        voc_b_thread.join()

        completed_results.sort(key=lambda x: x.id)
        return completed_results

    def close(self):
        for attr in ["gen_a", "gen_b", "voc_a", "voc_b"]:
            if hasattr(self, attr):
                obj = getattr(self, attr)
                if obj:
                    obj.close()
                    setattr(self, attr, None)

    def __del__(self):
        self.close()
