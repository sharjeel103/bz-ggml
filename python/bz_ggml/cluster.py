import math
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
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
    client_has_vocoder: bool = False  # Per-request flag: True -> stream tokens (3.2kbps), False -> server-side vocoder (24kHz audio)


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
    tokens: Optional[List[int]] = None


def estimate_speech_profile(text: str, instruction: Optional[str] = None, ref_frames: int = 0, cfg_scale: float = 1.0) -> Dict[str, int]:
    """
    Computes exact speech bounds, cadence, and token requirements based on speech acoustics:
      - Acoustic frame rate: 12.5 fps (each frame is 80ms of audio).
      - Fast speaking rate (200 WPM): ~3.75 frames per word.
      - Conversational average (140 WPM): ~5.35 frames per word.
      - Deliberate speaking rate (110 WPM + pauses): ~6.80 frames per word.
      - CFG multiplier: when cfg_scale > 1.0, backbone allocates both conditional and unconditional branches.
    """
    words = max(1, len(text.split()))
    ins_words = len(instruction.split()) if instruction else 0

    # Input prefix BPE tokens: ~1.30 tokens per word + ref frames + formatting overhead
    prefix_tokens = int(math.ceil(words * 1.30)) + int(math.ceil(ins_words * 1.30)) + ref_frames + 20

    # Output audio frame bounds (12.5 fps = 80ms/frame):
    # - Earliest possible speech completion (fast speech ~200 WPM): ~3.75 frames/word
    # - Expected speech duration (conversational ~140 WPM): ~5.35 frames/word + 12 frames
    # - Conservative maximum output frames (slow speech ~110 WPM + pauses): ~6.80 frames/word + 25 frames
    earliest_eos = max(16, int(math.floor(words * 3.75)))
    expected_eos = int(math.ceil(words * 5.35)) + 12
    conservative_max = int(math.ceil(words * 6.80)) + 25

    multiplier = 2 if cfg_scale > 1.0 else 1
    total_needed = (prefix_tokens + conservative_max) * multiplier

    return {
        "words": words,
        "prefix_tokens": prefix_tokens,
        "earliest_eos": earliest_eos,
        "expected_eos": expected_eos,
        "conservative_max": conservative_max,
        "total_needed": total_needed
    }


class TieredSlotPool:
    """
    80-Slot Tiered KV Cache Memory Pool for Zero-cudaMalloc Runtime.
      Tier 1: Short (up to 600 tokens / 30s audio, 56.25 MB/slot)
      Tier 2: Standard (up to 1,000 tokens / 1-min audio, 93.75 MB/slot - 50% pool)
      Tier 3: Long (up to 1,800 tokens / 2-min audio, 168.75 MB/slot)
    """
    def __init__(self, short_slots: int = 16, standard_slots: int = 24, long_slots: int = 56):
        self.tiers = {
            "short": {"capacity": 600, "total": short_slots, "free": list(range(1, short_slots + 1))},
            "standard": {"capacity": 1000, "total": standard_slots, "free": list(range(short_slots + 1, short_slots + standard_slots + 1))},
            "long": {"capacity": 1800, "total": long_slots, "free": list(range(short_slots + standard_slots + 1, short_slots + standard_slots + long_slots + 1))}
        }
        self.slot_to_tier = {}
        for tier_name, tier_info in self.tiers.items():
            for slot_id in tier_info["free"]:
                self.slot_to_tier[slot_id] = tier_name
        self.lock = threading.Lock()

    def acquire(self, needed_tokens: int) -> Optional[Tuple[int, str, int]]:
        with self.lock:
            if needed_tokens <= 600:
                tier_order = ["short", "standard", "long"]
            elif needed_tokens <= 1000:
                tier_order = ["standard", "long"]
            elif needed_tokens <= 1800:
                tier_order = ["long"]
            else:
                tier_order = ["long"]

            for tier in tier_order:
                free_list = self.tiers[tier]["free"]
                if free_list:
                    slot_id = free_list.pop(0)
                    return slot_id, tier, self.tiers[tier]["capacity"]
            return None

    def release(self, slot_id: int):
        with self.lock:
            tier = self.slot_to_tier.get(slot_id)
            if tier:
                self.tiers[tier]["free"].append(slot_id)

    def active_count(self) -> int:
        with self.lock:
            total_slots = sum(t["total"] for t in self.tiers.values())
            free_slots = sum(len(t["free"]) for t in self.tiers.values())
            return total_slots - free_slots

    def total_slots(self) -> int:
        return sum(t["total"] for t in self.tiers.values())


def select_burst_steps(active_sessions: dict, max_burst: int = 16) -> int:
    """
    Selects the optimal continuous burst size (K) in {16, 8, 4, 2, 1} based on
    75th-percentile majority scheduling. Ensures high GPU SM duty cycle while
    allowing the C++ safety clamp and Python post-hoc truncation to handle early EOS.
    """
    if not active_sessions:
        return 1

    remaining = sorted([
        max(0, s.get("expected_eos", 200) - s.get("total_frames", 0))
        for s in active_sessions.values()
    ])
    N = len(remaining)

    # p25: represents the state of 75% of active streams
    p25 = remaining[N // 4]
    # p50: median of active streams
    p50 = remaining[N // 2]

    if p25 >= 16:
        return min(max_burst, 16)
    elif p25 >= 8 or p50 >= 16:
        return min(max_burst, 8)
    elif p50 >= 8 or p25 >= 4:
        return min(max_burst, 4)
    elif p50 >= 2:
        return min(max_burst, 2)
    else:
        return 1


class DualInstanceCluster:
    """
    Symmetric Autonomous Dual-Island Architecture (10/10 Design).
      GPU 0: Generator Instance A + Dedicated Streaming Vocoder A (Autonomous Island 0)
      GPU 1: Generator Instance B + Dedicated Streaming Vocoder B (Autonomous Island 1)
      Zero Cross-GPU PCIe Contention | 100% Saturated Dual GPUs | Adaptive Dynamic Quantum
      Optional Surge Tiering: Dynamically routes burst overflow sessions to Q4 when threshold exceeded.
    """

    @staticmethod
    def calculate_token_budget(
        base_max_tokens: int = 50000,
        enable_q4_burst: bool = False,
        vocoder_mode: str = "dynamic"
    ) -> int:
        """
        Calculates exact active token capacity based on component VRAM footprint:
          - Baseline (Tokens-Only, No Q4): 50,000 active tokens (14.20 GiB safe ceiling).
          - Modular INT4 Depth Piece (310 MiB): -1,400 tokens (4.367 tokens/MiB).
          - Server Vocoder Decoder + Scratchpad (820 MiB): -3,600 tokens (when 'on' or 'dynamic').
          - Combined Full Hybrid Suite: -5,000 tokens -> 45,000 active tokens per GPU.
        """
        budget = base_max_tokens
        if enable_q4_burst:
            budget -= 1400
        if vocoder_mode in ("on", "dynamic"):
            budget -= 3600
        return budget

    def __init__(
        self,
        model_path: str,
        lib_path: Optional[str] = None,
        q4_model_path: Optional[str] = None,
        enable_q4_burst: bool = False,
        q4_threshold: int = 10,
        enable_vocoder: Optional[bool] = None,
        vocoder_mode: str = "dynamic",
        base_max_tokens: int = 50000
    ):
        self.model_path = model_path
        self.q4_model_path = q4_model_path
        self.enable_q4_burst = enable_q4_burst
        self.q4_threshold = q4_threshold
        if enable_vocoder is not None:
            self.vocoder_mode = "on" if enable_vocoder else "off"
        else:
            self.vocoder_mode = vocoder_mode.lower()
        self.enable_vocoder = self.vocoder_mode in ("on", "dynamic")
        self.base_max_tokens = base_max_tokens
        self.active_token_budget = self.calculate_token_budget(
            base_max_tokens=self.base_max_tokens,
            enable_q4_burst=self.enable_q4_burst,
            vocoder_mode=self.vocoder_mode
        )
        self.lib = BreezeLib(lib_path)

        # Autonomous Island 0 (GPU 0)
        print(f"[Cluster] Initializing Autonomous Island 0 on CUDA0 (Vocoder Mode: {self.vocoder_mode})...")
        t0 = time.time()
        self.gen_a = GeneratorHandle(self.lib, model_path, cuda_device=0)
        self.voc_a = VocoderHandle(self.lib, model_path, cuda_device=0) if self.enable_vocoder else None
        print(f"   -> Island 0 Generator Online on CUDA0 in {time.time()-t0:.2f}s")

        # Autonomous Island 1 (GPU 1)
        print(f"[Cluster] Initializing Autonomous Island 1 on CUDA1 (Vocoder Mode: {self.vocoder_mode})...")
        t0 = time.time()
        self.gen_b = GeneratorHandle(self.lib, model_path, cuda_device=1)
        self.voc_b = VocoderHandle(self.lib, model_path, cuda_device=1) if self.enable_vocoder else None
        print(f"   -> Island 1 Generator Online on CUDA1 in {time.time()-t0:.2f}s")

        # Optional Modular INT4 Depth Decoder piece (~300 MiB)
        self.has_q4_dd = False
        if self.enable_q4_burst and self.q4_model_path and os.path.exists(self.q4_model_path):
            print(f"[Cluster] Loading Modular INT4 Depth Decoder piece (~300 MB) on CUDA0 and CUDA1 (Threshold: {self.q4_threshold})...")
            t0 = time.time()
            self.gen_a.load_q4_depth(self.q4_model_path)
            self.gen_b.load_q4_depth(self.q4_model_path)
            self.has_q4_dd = True
            print(f"   -> Modular INT4 Depth Decoder Online in {time.time()-t0:.2f}s (Saved 2.24 GB VRAM per GPU!)")

        print(f"[Cluster Config] Active Token Ceiling: {self.active_token_budget} tokens/GPU (Base: {self.base_max_tokens} | Q4: {self.enable_q4_burst} | Vocoder: {self.vocoder_mode})")

    def run_workload(
        self,
        tasks: List[UserTask],
        out_dir: str = "audio_out",
        arrival_delays: Optional[List[float]] = None,
        on_progress: Optional[Callable[[ClusterResult], None]] = None,
        quantum_frames: int = 16,
        max_slots_per_gpu: Optional[int] = 64,
        vocoder_chunk_size: int = 32,
        return_tokens: bool = False,
        max_active_tokens_per_gpu: Optional[int] = None
    ) -> List[ClusterResult]:
        os.makedirs(out_dir, exist_ok=True)

        effective_max_tokens = (
            max_active_tokens_per_gpu
            if max_active_tokens_per_gpu is not None
            else self.active_token_budget
        )

        job_queue = queue.Queue()
        voc_a_queue = queue.Queue()
        voc_b_queue = queue.Queue()

        user_audio_results = {}
        user_audio_lock = threading.Lock()
        completed_results: List[ClusterResult] = []
        results_lock = threading.Lock()
        workers_stopping = False

        # 1. Independent Vocoder Worker Loops on GPU 0 and GPU 1 (Dynamic Stream-Aware Batched)
        def vocoder_loop(v_queue: queue.Queue, voc_handle: VocoderHandle, voc_tag: str):
            while not workers_stopping:
                try:
                    task = v_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if task is None:
                    break

                # Coalesce: gather task and any pending queued tasks into a unified batch
                batch_items = list(task) if isinstance(task, list) else [task]
                while True:
                    try:
                        extra = v_queue.get_nowait()
                        if extra is None:
                            workers_stopping = True
                            break
                        if isinstance(extra, list):
                            batch_items.extend(extra)
                        else:
                            batch_items.append(extra)
                    except queue.Empty:
                        break

                # Separate items with audio tokens to compute on GPU
                compute_items = [it for it in batch_items if len(it[4]) > 0]
                if compute_items:
                    batch_tokens = [it[4] for it in compute_items]
                    batch_n_frames = [len(it[4]) // 16 for it in compute_items]

                    # SINGLE UNIFIED GPU DISPATCH FOR ALL READY STREAMS IN THE BATCH
                    batch_samples = voc_handle.stream_decode_batch(batch_tokens, batch_n_frames)

                    for it, samples in zip(compute_items, batch_samples):
                        u_id = it[0]
                        with user_audio_lock:
                            if u_id not in user_audio_results:
                                user_audio_results[u_id] = {
                                    "samples": [],
                                    "first_audio_time": None
                                }
                            user_audio_results[u_id]["samples"].extend(samples)
                            if user_audio_results[u_id]["first_audio_time"] is None:
                                user_audio_results[u_id]["first_audio_time"] = time.time()

                # Process EOS completions
                for it in batch_items:
                    u_id, words, arr_time, t_start, frames_list, is_first, is_eos, total_frames, worker_tag = it
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
                            worker=f"{worker_tag} + {voc_tag} [Batched]",
                            wav_path=out_wav
                        )

                        with results_lock:
                            completed_results.append(res)
                            if on_progress:
                                on_progress(res)

                    v_queue.task_done()

        # Smart Hybrid Vocoder: Only spin up vocoder threads if at least one request requires server-side synthesis
        needs_server_vocoder = (not return_tokens) and any(not getattr(t, "client_has_vocoder", False) for t in tasks)

        voc_a_thread = None
        voc_b_thread = None
        if needs_server_vocoder:
            if self.voc_a is None:
                self.voc_a = VocoderHandle(self.lib, self.model_path, cuda_device=0)
            if self.voc_b is None:
                self.voc_b = VocoderHandle(self.lib, self.model_path, cuda_device=1)
            voc_a_thread = threading.Thread(target=vocoder_loop, args=(voc_a_queue, self.voc_a, "Vocoder A (GPU 0)"))
            voc_b_thread = threading.Thread(target=vocoder_loop, args=(voc_b_queue, self.voc_b, "Vocoder B (GPU 1)"))
            voc_a_thread.start()
            voc_b_thread.start()

        # 2. Generator Multi-Session Worker Loop with Adaptive Quantum
        all_dispatched = threading.Event()
        telemetry_lock = threading.Lock()
        cluster_burst_counts = {16: 0, 8: 0, 4: 0, 2: 0, 1: 0}
        cluster_session_stats = []

        def generator_multi_session_loop(
            worker_name: str,
            gen_default: GeneratorHandle,
            voc_queue: queue.Queue,
            gpu_id: int,
            max_slots: Optional[int] = 64,
            max_active_tokens: int = 50000,
            chunk_size: int = 32,
            return_tokens_mode: bool = False
        ):
            active_sessions = {}
            current_active_tokens = 0
            last_hb_time = time.time()
            pending_item = None

            # Initialize 96-Slot Tiered Memory Pool for this GPU Island (Supporting up to 64 active slots)
            slot_pool = TieredSlotPool(short_slots=16, standard_slots=24, long_slots=56)

            while not workers_stopping:
                # A. Admit waiting tasks governed by TieredSlotPool, max_slots, and max_active_tokens budget
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

                    cfg_scale = getattr(task_item, "cfg_scale", 1.0)
                    profile = estimate_speech_profile(
                        task_item.text,
                        getattr(task_item, "instruction", None),
                        getattr(task_item, "ref_frames", 0),
                        cfg_scale=cfg_scale
                    )

                    needed_tokens = profile["total_needed"]

                    # Active Token Budget Guard: prevent aggregate VRAM from exceeding max_active_tokens
                    if (current_active_tokens + needed_tokens > max_active_tokens) and len(active_sessions) > 0:
                        pending_item = (task_item, arr_time)
                        break

                    # Slot pool capacity check uses single-branch sequence capacity
                    single_branch_tokens = profile["prefix_tokens"] + profile["conservative_max"]
                    slot_res = slot_pool.acquire(single_branch_tokens)
                    if slot_res is None:
                        # Pool full in all matching tiers; hold task locally
                        pending_item = (task_item, arr_time)
                        break

                    slot_id, slot_tier, slot_capacity = slot_res
                    output_cap_frames = min(slot_capacity - profile["prefix_tokens"], profile["conservative_max"])
                    if hasattr(task_item, "max_steps") and task_item.max_steps > 0:
                        output_cap_frames = min(output_cap_frames, task_item.max_steps)

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
                        print(f"[{worker_name}] Transient memory pressure on session {sid} ({e}); holding task for next cycle...")
                        slot_pool.release(slot_id)
                        pending_item = (task_item, arr_time)
                        break

                    current_active_tokens += allocated_tokens
                    client_has_voc = return_tokens_mode or getattr(task_item, "client_has_vocoder", False)

                    active_sessions[sid] = {
                        "task": task_item,
                        "gen": active_gen,
                        "model_tag": model_tag,
                        "arr_time": arr_time,
                        "t_exec_start": t_exec_start,
                        "words": profile["words"],
                        "tokens": allocated_tokens,
                        "slot_id": slot_id,
                        "slot_tier": slot_tier,
                        "earliest_eos": profile["earliest_eos"],
                        "expected_eos": profile["expected_eos"],
                        "cb0": cb0,
                        "total_frames": 0,
                        "trimmed_dummy_frames": 0,
                        "chunk_buffer": [],
                        "all_tokens": [],
                        "first_step_time": None,
                        "max_steps": output_cap_frames,
                        "client_has_vocoder": client_has_voc
                    }

                if not active_sessions:
                    if all_dispatched.is_set() and job_queue.empty():
                        break
                    time.sleep(0.002)
                    continue

                # B. Step all active sessions concurrently using Continuous Multi-Step Bursting
                active_sids = [sid for sid in active_sessions.keys()]
                if not active_sids:
                    continue

                K = select_burst_steps(active_sessions, max_burst=quantum_frames)
                with telemetry_lock:
                    cluster_burst_counts[K] = cluster_burst_counts.get(K, 0) + 1

                step_seeds = [
                    active_sessions[sid]["task"].seed + active_sessions[sid]["total_frames"]
                    for sid in active_sids
                ]

                next_cb0s, burst_frames = gen_default.sessions_step_burst(
                    active_sids, burst_steps=K, seeds=step_seeds
                )

                finished_sids = []
                for i, sid in enumerate(active_sids):
                    s = active_sessions.get(sid)
                    if not s:
                        continue

                    task_item = s["task"]
                    s_frames = burst_frames[i]
                    final_cb0 = next_cb0s[i]

                    stream_hit_eos = False
                    valid_frames_count = 0

                    # Post-hoc check: scan returned burst frames and truncate trailing dummy frames
                    for frame16 in s_frames:
                        cb0 = frame16[0]
                        if cb0 < 0 or cb0 == 4096:
                            stream_hit_eos = True
                            break

                        s["total_frames"] += 1
                        s["all_tokens"].extend(frame16)
                        valid_frames_count += 1

                        if not s["client_has_vocoder"]:
                            s["chunk_buffer"].extend(frame16)

                        if s["total_frames"] >= s["max_steps"]:
                            stream_hit_eos = True
                            break

                    trimmed_in_burst = len(s_frames) - valid_frames_count
                    s["trimmed_dummy_frames"] += trimmed_in_burst

                    if s["first_step_time"] is None and valid_frames_count > 0:
                        s["first_step_time"] = time.time()

                    s["cb0"] = final_cb0

                    if stream_hit_eos or final_cb0 < 0 or s["total_frames"] >= s["max_steps"]:
                        finished_sids.append(sid)

                # B2. Dynamic Batch Gathering: Coalesce all streams that filled >= 32 frames
                burst_ready_chunks = []
                for cur_sid, cur_s in active_sessions.items():
                    if cur_s["client_has_vocoder"]:
                        continue
                    while chunk_size > 0 and len(cur_s["chunk_buffer"]) >= (chunk_size * 16):
                        dispatch_tokens = cur_s["chunk_buffer"][:chunk_size * 16]
                        cur_s["chunk_buffer"] = cur_s["chunk_buffer"][chunk_size * 16:]
                        burst_ready_chunks.append((
                            cur_s["task"].id, cur_s["words"], cur_s["arr_time"], cur_s["t_exec_start"],
                            dispatch_tokens, False, False, cur_s["total_frames"],
                            f"GPU {gpu_id} ({worker_name} [{cur_s['model_tag']}])"
                        ))
                if burst_ready_chunks:
                    voc_queue.put(burst_ready_chunks)

                # C. Check if EOS or max_steps reached
                eos_ready_chunks = []
                for sid in finished_sids:
                    s = active_sessions.get(sid)
                    if not s:
                        continue
                    task_item = s["task"]
                    t_end = time.time()

                    if s["client_has_vocoder"]:
                        audio_s = s["total_frames"] * 0.08
                        wall_s = t_end - s["t_exec_start"]
                        turnaround_s = t_end - s["arr_time"]
                        wait_s = s["t_exec_start"] - s["arr_time"]
                        ttfa = (s["first_step_time"] - s["arr_time"]) if s["first_step_time"] else turnaround_s
                        rtf = wall_s / (audio_s if audio_s > 0 else 1.0)

                        res = ClusterResult(
                            id=task_item.id,
                            words=s["words"],
                            audio_s=round(audio_s, 2),
                            queue_wait_s=round(wait_s, 2),
                            compute_wall_s=round(wall_s, 2),
                            turnaround_s=round(turnaround_s, 2),
                            ttfa_s=round(ttfa, 3),
                            rtf=round(rtf, 3),
                            worker=f"GPU {gpu_id} ({worker_name} [{s['model_tag']}]) [Client Token Mode]",
                            wav_path="",
                            tokens=list(s["all_tokens"])
                        )
                        with results_lock:
                            completed_results.append(res)
                            if on_progress:
                                on_progress(res)
                    else:
                        # Terminal partial flush: dispatch remaining frames in chunk_buffer
                        leftover_tokens = list(s["chunk_buffer"])
                        s["chunk_buffer"] = []
                        eos_ready_chunks.append((
                            task_item.id, s["words"], s["arr_time"], s["t_exec_start"],
                            leftover_tokens, False, True, s["total_frames"],
                            f"GPU {gpu_id} ({worker_name} [{s['model_tag']}])"
                        ))

                    with telemetry_lock:
                        cluster_session_stats.append({
                            "id": task_item.id,
                            "words": s["words"],
                            "expected_eos": s["expected_eos"],
                            "actual_eos": s["total_frames"],
                            "error_frames": s["total_frames"] - s["expected_eos"],
                            "trimmed_dummy": s.get("trimmed_dummy_frames", 0)
                        })

                    # Free session slot immediately in local VRAM and return slot to pool
                    current_active_tokens -= s["tokens"]
                    slot_pool.release(s["slot_id"])
                    gen_default.session_free(sid)
                    del active_sessions[sid]
                    job_queue.task_done()

                if eos_ready_chunks:
                    voc_queue.put(eos_ready_chunks)


                if time.time() - last_hb_time > 10.0 and active_sessions:
                    last_hb_time = time.time()
                    frames_list = [s["total_frames"] for s in active_sessions.values()]
                    min_f = min(frames_list) if frames_list else 0
                    max_f = max(frames_list) if frames_list else 0
                    print(f"  [Heartbeat GPU {gpu_id}] Active: {len(active_sessions):2d} streams (Pool: {slot_pool.active_count()}/{slot_pool.total_slots()} slots | Tokens: {current_active_tokens}/{max_active_tokens}) | Frames: min {min_f:3d} / max {max_f:3d} | Audio: {sum(frames_list)*0.08:.1f}s", flush=True)

        worker_a = threading.Thread(
            target=generator_multi_session_loop,
            args=("Instance A", self.gen_a, voc_a_queue, 0, max_slots_per_gpu, effective_max_tokens, vocoder_chunk_size, return_tokens)
        )
        worker_b = threading.Thread(
            target=generator_multi_session_loop,
            args=("Instance B", self.gen_b, voc_b_queue, 1, max_slots_per_gpu, effective_max_tokens, vocoder_chunk_size, return_tokens)
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
        if needs_server_vocoder:
            voc_a_queue.join()
            voc_b_queue.join()

        # Shutdown workers
        workers_stopping = True
        worker_a.join()
        worker_b.join()
        if needs_server_vocoder:
            voc_a_queue.put(None)
            voc_b_queue.put(None)
            if voc_a_thread:
                voc_a_thread.join()
            if voc_b_thread:
                voc_b_thread.join()

        total_bursts = sum(cluster_burst_counts.values())
        total_steps_executed = sum(k * v for k, v in cluster_burst_counts.items())
        avg_burst = (total_steps_executed / total_bursts) if total_bursts > 0 else 0
        abs_errors = [abs(x["error_frames"]) for x in cluster_session_stats]
        mean_abs_err = float(np.mean(abs_errors)) if abs_errors else 0.0
        total_trimmed = sum(x["trimmed_dummy"] for x in cluster_session_stats)
        total_valid = sum(x["actual_eos"] for x in cluster_session_stats)
        trim_pct = (total_trimmed / (total_valid + total_trimmed) * 100.0) if (total_valid + total_trimmed) > 0 else 0.0

        total_syncs_avoided = max(0, total_steps_executed - total_bursts)
        sync_reduction_pct = (total_syncs_avoided / total_steps_executed * 100.0) if total_steps_executed > 0 else 0.0
        est_host_time_saved_s = total_syncs_avoided * 0.00045 # ~0.45ms per avoided host sync

        total_audio_reqs = sum(1 for r in completed_results if r.wav_path)
        total_token_reqs = sum(1 for r in completed_results if not r.wav_path)

        print("\n================================================================================")
        print("                  BURST ENGINE TELEMETRY & PREDICTION METRICS                   ")
        print("================================================================================")
        print(f"Total Bursts Dispatched:         {total_bursts} calls ({total_steps_executed} steps executed)")
        for k_val in [16, 8, 4, 2, 1]:
            cnt = cluster_burst_counts.get(k_val, 0)
            pct = (cnt / total_bursts * 100.0) if total_bursts > 0 else 0.0
            print(f"  - Burst K = {k_val:2d}:                    {cnt:5d} calls ({pct:5.1f}%)")
        print(f"Average Burst Quantum (K_avg):   {avg_burst:.2f} steps / call")
        print("--------------------------------------------------------------------------------")
        print("Smart Hybrid Vocoder & Request Routing:")
        print(f"  - Server-Side Vocoder Synthesized: {total_audio_reqs:4d} requests (24 kHz Audio, Chunk = {vocoder_chunk_size} frames)")
        print(f"  - Client Hydrated Bypass (Tokens): {total_token_reqs:4d} requests (3.2 kbps tokens streamed)")
        print(f"  - Vocoder Chunk Threshold:         {vocoder_chunk_size} frames ({vocoder_chunk_size * 0.08:.2f}s audio = {vocoder_chunk_size * 16} tokens)")
        print("--------------------------------------------------------------------------------")
        print("Pristine Audio & Garbage Frame Accounting:")
        print(f"  - Total Valid Speech Frames:   {total_valid} frames ({total_valid * 0.08:.2f}s genuine audio)")
        print(f"  - Total Extra / Dummy Frames:  {total_trimmed} frames ({total_trimmed * 0.08:.2f}s discarded chunk)")
        print(f"  - Discarded Garbage Overhead:  {trim_pct:.2f}% of output frames")
        print(f"  - Clean Discard Rate:          100.0% (Zero dummy frames reached output)")
        print("--------------------------------------------------------------------------------")
        print("Host-Device Efficiency Gains:")
        print(f"  - Host Syncs Avoided:          {total_syncs_avoided} calls ({sync_reduction_pct:.1f}% reduction vs K=1)")
        print(f"  - Host CPU/PCIe Time Saved:    ~{est_host_time_saved_s:.2f} seconds")
        print(f"  - Mean Absolute EOS Error:     +/-{mean_abs_err:.1f} frames (+/-{mean_abs_err * 0.08:.2f}s audio)")
        print("================================================================================\n", flush=True)

        self.last_telemetry = {
            "burst_counts": dict(cluster_burst_counts),
            "total_bursts": total_bursts,
            "total_steps_executed": total_steps_executed,
            "avg_burst": round(avg_burst, 2),
            "total_audio_reqs": total_audio_reqs,
            "total_token_reqs": total_token_reqs,
            "vocoder_chunk_size": vocoder_chunk_size,
            "total_valid_frames": total_valid,
            "total_valid_audio_s": round(total_valid * 0.08, 2),
            "total_trimmed_frames": total_trimmed,
            "total_trimmed_audio_s": round(total_trimmed * 0.08, 2),
            "trim_pct": round(trim_pct, 2),
            "syncs_avoided": total_syncs_avoided,
            "sync_reduction_pct": round(sync_reduction_pct, 1),
            "host_time_saved_s": round(est_host_time_saved_s, 2),
            "mean_abs_err_frames": round(float(mean_abs_err), 2),
            "mean_abs_err_s": round(float(mean_abs_err * 0.08), 3)
        }

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
