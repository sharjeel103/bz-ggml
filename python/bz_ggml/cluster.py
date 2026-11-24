import itertools
import math
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .bindings import (
    BreezeLib,
    GeneratorHandle,
    VocoderHandle,
    serialize_breeze_voice,
    deserialize_breeze_voice,
)


@dataclass
class UserTask:
    id: int
    text: str
    instruction: str = "Speak clearly and naturally."
    cfg_scale: float = 1.0
    temperature: float = 0.60
    top_p: float = 0.88
    top_k: int = 40
    repetition_penalty: float = 1.1
    depth_temperature: float = 0.0
    depth_cfg_codebooks: int = 15
    seed: int = 42
    max_steps: int = 2040
    ref_text: Optional[str] = None
    ref_codes: Optional[List[int]] = None
    ref_frames: int = 0
    client_has_vocoder: bool = False
    stream_pcm: bool = False  # Non-streaming by default


@dataclass
class VoiceConversionTask:
    id: int
    src_codes: List[int]
    src_frames: int
    ref_text: Optional[str] = None
    ref_codes: Optional[List[int]] = None
    ref_frames: int = 0
    ref_audio: Optional[List[float]] = None
    src_text: Optional[str] = None
    cfg_scale: float = 1.5
    keep_acoustic: int = 0
    feed_source: bool = True
    seed: int = 42
    client_has_vocoder: bool = False
    stream_pcm: bool = False


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
    success: bool = True
    error_msg: Optional[str] = None

    @property
    def audio_path(self) -> str:
        return self.wav_path


def estimate_conversion_profile(
    src_frames: int,
    ref_frames: int = 0,
    ref_text: Optional[str] = None,
    src_text: Optional[str] = None,
    cfg_scale: float = 1.5,
) -> Dict[str, Any]:
    """Computes exact deterministic token requirements for Mode 4 Voice Conversion."""
    ref_text_words = len(ref_text.split()) if ref_text else 0
    ref_text_tokens = int(math.ceil(ref_text_words * 1.30)) if ref_text else 0

    if src_text:
        src_text_words = len(src_text.split())
        text_tokens = int(math.ceil(src_text_words * 1.30))
    else:
        secs = src_frames / 12.5
        text_tokens = int(math.ceil(secs * 4.0)) + 4

    total_c = 1 + ref_text_tokens + (ref_frames + 1 if ref_frames > 0 else 0) + 1 + text_tokens
    st_c_capacity = total_c + src_frames + 8

    use_cfg = cfg_scale > 1.0
    if use_cfg:
        total_u = 1 + text_tokens
        st_u_capacity = total_u + src_frames + 8
        total_needed = st_c_capacity + st_u_capacity
    else:
        st_u_capacity = 0
        total_needed = st_c_capacity

    return {
        "src_frames": src_frames,
        "ref_frames": ref_frames,
        "text_tokens": text_tokens,
        "st_c_capacity": st_c_capacity,
        "st_u_capacity": st_u_capacity,
        "total_needed": total_needed,
        "vram_estimate_mib": (total_needed * 229376) / (1024 * 1024),
    }


class ConversionWorkerPool:
    """Strict concurrency limiter for Voice Conversion and Audio Reference Encoding."""

    def __init__(self, max_concurrent: int = 2):
        self.semaphore = threading.BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self.lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        acquired = self.semaphore.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            with self.lock:
                self.active_count += 1
        return acquired

    def release(self):
        with self.lock:
            self.active_count = max(0, self.active_count - 1)
        self.semaphore.release()


def estimate_speech_profile(
    text: str,
    instruction: Optional[str] = None,
    ref_frames: int = 0,
    cfg_scale: float = 1.0,
    ref_text: Optional[str] = None,
) -> Dict[str, int]:
    """
    Computes exact speech bounds, cadence, and KV-cache token requirements:
      - Acoustic frame rate: 12.5 fps (80 ms/frame).
      - Fast (~200 WPM): ~3.75 frames/word.
      - Conversational (~140 WPM): ~5.35 frames/word.
      - Deliberate (~110 WPM + pauses): ~6.80 frames/word.
      - CFG multiplier: when cfg_scale != 1.0, backbone allocates both cond & uncond branches.
    """
    words = max(1, len(text.split()))
    ins_words = len(instruction.split()) if instruction else 0
    ref_words = len(ref_text.split()) if ref_text else 0

    prefix_tokens = (
        int(math.ceil(words * 1.30))
        + int(math.ceil(ins_words * 1.30))
        + int(math.ceil(ref_words * 1.30))
        + ref_frames
        + 20
    )

    earliest_eos = max(16, int(math.floor(words * 3.75)))
    expected_eos = int(math.ceil(words * 5.35)) + 12
    conservative_max = int(math.ceil(words * 6.80)) + 25

    multiplier = 2 if abs(cfg_scale - 1.0) > 1e-4 else 1
    single_branch_total = min(2040, prefix_tokens + conservative_max)
    total_needed = single_branch_total * multiplier

    return {
        "words": words,
        "prefix_tokens": prefix_tokens,
        "earliest_eos": earliest_eos,
        "expected_eos": expected_eos,
        "conservative_max": conservative_max,
        "single_branch_total": single_branch_total,
        "total_needed": total_needed,
    }


def select_burst_steps(active_sessions: dict, max_burst: int = 16) -> int:
    """
    Selects continuous burst size K in {16, 8, 4, 2, 1} using 75th-percentile majority
    scheduling so short streams finish promptly without starving large batches.
    """
    if not active_sessions:
        return 1

    remaining = sorted(
        max(0, s.get("expected_eos", 200) - s.get("total_frames", 0))
        for s in active_sessions.values()
    )
    n = len(remaining)
    p25 = remaining[n // 4]
    p50 = remaining[n // 2]

    if p25 >= 16:
        return min(max_burst, 16)
    elif p25 >= 8 or p50 >= 16:
        return min(max_burst, 8)
    elif p50 >= 8 or p25 >= 4:
        return min(max_burst, 4)
    elif p50 >= 2:
        return min(max_burst, 2)
    return 1


class TokenCapacityCoordinator:
    """
    Intelligent hardware-aware token capacity coordinator for Dual Tesla T4 cluster.
    Guarantees exactly ~500 MiB safety headroom at peak load across all configurations
    (INT8, FP16, Q4 burst, unconstrained ASR, and on-demand Enhancer).
    """

    USABLE_VRAM_MIB = 14910
    SAFETY_HEADROOM_MIB = 500
    TARGET_MAX_VRAM_MIB = 14410
    VOCODER_SCRATCHPAD_MIB = 1270
    ASR_PEAK_MIB = 2160
    ENHANCER_PEAK_MIB = 5055
    TOKENS_PER_MIB = 4

    def __init__(
        self,
        precision: str = "int8",
        enable_q4_burst: bool = False,
        enable_asr: bool = True,
        enable_enhancer: bool = True,
    ):
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.active_tokens = {0: 0, 1: 0}
        self.enhancer_active = False

        # 1. Base Generator Footprint (INT8 / FP16 / Q4 burst)
        gen_vram = 3400
        if precision.lower() in ("fp16", "float16"):
            gen_vram += 3400
        if enable_q4_burst:
            gen_vram += 1400
        self.gen_vram = gen_vram

        # 2. GPU 0 Fixed Ceiling (30,000 tokens for INT8)
        static_0 = self.gen_vram + self.VOCODER_SCRATCHPAD_MIB + (self.ASR_PEAK_MIB if enable_asr else 0)
        kv_0 = max(1000, self.TARGET_MAX_VRAM_MIB - static_0)
        self.ceiling_gpu0 = min(30000, (kv_0 * self.TOKENS_PER_MIB // 100) * 100)

        # 3. GPU 1 Normal Ceiling (38,900 tokens for INT8 - Enhancer is 0 MiB on GPU when idle)
        static_1 = self.gen_vram + self.VOCODER_SCRATCHPAD_MIB
        kv_1_normal = max(1000, self.TARGET_MAX_VRAM_MIB - static_1)
        self.ceiling_gpu1_normal = min(38900, (kv_1_normal * self.TOKENS_PER_MIB // 100) * 100)

        # 4. GPU 1 Enhancer Clamped Ceiling (18,400 tokens for INT8 - accounts for all 5,055 MiB weights + inference)
        kv_1_enh = max(1000, self.TARGET_MAX_VRAM_MIB - (static_1 + self.ENHANCER_PEAK_MIB))
        self.ceiling_gpu1_enh = min(18400, (kv_1_enh * self.TOKENS_PER_MIB // 100) * 100)

    def get_admission_ceiling(self, gpu_id: int) -> int:
        with self.lock:
            if gpu_id == 0:
                return self.ceiling_gpu0
            return self.ceiling_gpu1_enh if self.enhancer_active else self.ceiling_gpu1_normal

    def get_free_headroom(self, gpu_id: int) -> int:
        with self.lock:
            return self.get_admission_ceiling(gpu_id) - self.active_tokens.get(gpu_id, 0)

    def update_active_tokens(self, gpu_id: int, current_tokens: int):
        with self.lock:
            self.active_tokens[gpu_id] = max(0, current_tokens)
            self.cv.notify_all()

    def acquire_enhancer_headroom(self) -> bool:
        """
        Pure event-driven headroom acquisition.
        Zero fixed sleep intervals, zero polling.
        Wakes up immediately via Condition notification the exact microsecond
        active tokens on GPU 1 drop to or below the Enhancer ceiling.
        """
        with self.lock:
            self.enhancer_active = True
            target = self.ceiling_gpu1_enh
            while self.active_tokens.get(1, 0) > target:
                self.cv.wait()
            return True

    def release_enhancer_headroom(self):
        with self.lock:
            self.enhancer_active = False
            self.cv.notify_all()

    def request_headroom(self, gpu_id: int, tokens_needed: int = 0, timeout: float = 0.0) -> bool:
        if gpu_id == 1:
            return self.acquire_enhancer_headroom()
        return True

    def release_headroom(self, gpu_id: int, tokens_needed: int = 0):
        if gpu_id == 1:
            self.release_enhancer_headroom()


DynamicCapacityCoordinator = TokenCapacityCoordinator


class DualInstanceCluster:
    """
    Persistent Dual-T4 Autonomous Cluster.
      - GPU 0: Generator Island A + Shared Tiled Vocoder A
      - GPU 1: Generator Island B + Shared Tiled Vocoder B
      - Two persistent generator loops continuously pull TTS chunks from a shared queue,
        providing true cross-request and cross-chunk continuous batching with zero
        per-chunk thread churn, globally unique session IDs, and bounded tiled vocoding.
    """

    @staticmethod
    def calculate_token_budget(
        base_max_tokens: int = 38400,
        enable_q4_burst: bool = False,
        vocoder_mode: str = "dynamic",
        precision: str = "int8",
        enable_enhancer: bool = False,
        enable_asr: bool = False,
        **kwargs,
    ) -> int:
        budget = base_max_tokens
        if precision.lower() in ("fp16", "float16"):
            budget -= 13600
        if enable_q4_burst:
            budget -= 5600
        return max(4000, budget)

    def __init__(
        self,
        model_path: str,
        lib_path: Optional[str] = None,
        q4_model_path: Optional[str] = None,
        enable_q4_burst: bool = False,
        q4_threshold: int = 10,
        enable_vocoder: Optional[bool] = None,
        vocoder_mode: str = "on",
        precision: str = "int8",
        enable_enhancer: bool = True,
        enable_asr: bool = True,
        base_max_tokens: int = 38400,
        max_slots_per_gpu: int = 48,
        quantum_frames: int = 16,
    ):
        self.model_path = model_path
        self.q4_model_path = q4_model_path
        self.enable_q4_burst = enable_q4_burst
        self.q4_threshold = q4_threshold
        self.precision = precision
        self.enable_enhancer = enable_enhancer
        self.enable_asr = enable_asr
        self.sample_rate = 24000
        self.max_slots_per_gpu = max_slots_per_gpu
        self.quantum_frames = quantum_frames

        if enable_vocoder is not None:
            self.vocoder_mode = "on" if enable_vocoder else "off"
        else:
            self.vocoder_mode = vocoder_mode.lower()
        self.enable_vocoder = self.vocoder_mode in ("on", "dynamic")
        self.base_max_tokens = base_max_tokens
        self.capacity_coord = TokenCapacityCoordinator(
            precision=self.precision,
            enable_q4_burst=self.enable_q4_burst,
            enable_asr=self.enable_asr,
            enable_enhancer=self.enable_enhancer,
        )
        self.dynamic_coord = self.capacity_coord
        self.active_token_budget = self.capacity_coord.ceiling_gpu1_normal
        self.lib = BreezeLib(lib_path)
        self._voc_rr_index = 0
        self._voc_rr_lock = threading.Lock()

        # Island 0 (GPU 0)
        print(f"[Cluster] Initializing Island 0 on CUDA0 (Vocoder: {self.vocoder_mode})...")
        t0 = time.time()
        self.gen_a = GeneratorHandle(self.lib, model_path, cuda_device=0)
        self.voc_a = self.gen_a.create_vocoder() if self.enable_vocoder else None
        self._voc_lock_a = threading.Lock()
        print(f"   -> Island 0 Online on CUDA0 in {time.time() - t0:.2f}s")

        # Island 1 (GPU 1)
        print(f"[Cluster] Initializing Island 1 on CUDA1 (Vocoder: {self.vocoder_mode})...")
        t0 = time.time()
        self.gen_b = GeneratorHandle(self.lib, model_path, cuda_device=1)
        self.voc_b = self.gen_b.create_vocoder() if self.enable_vocoder else None
        self._voc_lock_b = threading.Lock()
        self.is_enhancer_running = False
        print(f"   -> Island 1 Online on CUDA1 in {time.time() - t0:.2f}s")

        # Warm up vocoder buffers upfront so the 1,266 MiB gallocr scratchpad is pre-allocated
        if self.enable_vocoder:
            print("[Cluster] Pre-allocating bounded tiled vocoder scratchpads on CUDA0 & CUDA1...")
            dummy_tokens = [0] * (16 * 240)
            if self.voc_a:
                try:
                    self.voc_a.decode_tiled(dummy_tokens, ctx_frames=80, step_frames=240)
                except Exception as e:
                    print(f"   -> Vocoder A warmup warning: {e}")
            if self.voc_b:
                try:
                    self.voc_b.decode_tiled(dummy_tokens, ctx_frames=80, step_frames=240)
                except Exception as e:
                    print(f"   -> Vocoder B warmup warning: {e}")
            print("   -> Bounded Tiled Vocoder Scratchpads Pre-Allocated Cleanly.")

        # Optional Modular INT4 Depth Decoder (~300 MiB)
        self.has_q4_dd = False
        if self.enable_q4_burst and self.q4_model_path and os.path.exists(self.q4_model_path):
            print(f"[Cluster] Loading Modular INT4 Depth Decoder on CUDA0 & CUDA1 (Threshold: {self.q4_threshold})...")
            t0 = time.time()
            self.gen_a.load_q4_depth(self.q4_model_path)
            self.gen_b.load_q4_depth(self.q4_model_path)
            self.has_q4_dd = True
            print(f"   -> Modular INT4 Depth Decoder Online in {time.time() - t0:.2f}s")

        # Concurrency Limiters for Voice Conversion & Audio Reference Encoding
        self.max_concurrent_conversions = 2
        self.conv_pool_a = ConversionWorkerPool(max_concurrent=self.max_concurrent_conversions)
        self.conv_pool_b = ConversionWorkerPool(max_concurrent=self.max_concurrent_conversions)
        self.audio_encoder_pool = ConversionWorkerPool(max_concurrent=2)

        # Globally unique session ID counter (eliminates cross-request session collisions)
        self._sid_counter = itertools.count(1)
        self._sid_lock = threading.Lock()

        # Persistent dual-GPU TTS job queue & worker threads
        self._tts_queue: queue.Queue = queue.Queue()
        self._running = True
        self._worker_a = threading.Thread(
            target=self._persistent_generator_loop,
            args=("Instance A", self.gen_a, 0),
            daemon=True,
            name="ClusterGen-GPU0",
        )
        self._worker_b = threading.Thread(
            target=self._persistent_generator_loop,
            args=("Instance B", self.gen_b, 1),
            daemon=True,
            name="ClusterGen-GPU1",
        )
        self._worker_a.start()
        self._worker_b.start()

        print(
            f"[Cluster Config] Persistent Dual-GPU Engine Online | "
            f"Active Token Ceiling: {self.active_token_budget} tokens/GPU | "
            f"Max Slots/GPU: {self.max_slots_per_gpu} | Vocoder: {self.vocoder_mode}"
        )

    def _next_session_id(self) -> int:
        with self._sid_lock:
            return next(self._sid_counter)

    def submit_tts_chunk(
        self,
        task: UserTask,
        on_complete: Callable[[Optional[List[int]], Optional[str], Dict[str, Any]], None],
        cancel_event: Optional[threading.Event] = None,
    ) -> int:
        """
        Non-blocking submission of a single TTS chunk to the persistent dual-GPU cluster.
        Returns a globally unique session_id. When generation finishes, invokes:
            on_complete(tokens: Optional[List[int]], error_msg: Optional[str], stats: Dict)
        directly in memory with zero disk writes.
        """
        if not self._running:
            raise RuntimeError("DualInstanceCluster is shut down.")
        sid = self._next_session_id()
        self._tts_queue.put((sid, task, time.time(), on_complete, cancel_event))
        return sid

    def decode_tokens_tiled(
        self,
        tokens: List[int],
        ctx_frames: int = 80,
        step_frames: int = 240,
        gpu_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decodes an arbitrary-length 16-codebook token array into 24 kHz float32 PCM using
        strictly VRAM-bounded tiled decoding.
        Primary vocoder host: GPU 0 (always available, Whisper leaves >3.7 GB free).
        Secondary / spillover host: GPU 1 (runs when GPU 0 is busy and is_enhancer_running is False).
        """
        if not tokens:
            return np.zeros(0, dtype=np.float32)

        # Equal-priority dynamic load balancing between GPU 0 and GPU 1
        if gpu_id in (0, 1):
            target_gpu = gpu_id
        else:
            lock_a_busy = self._voc_lock_a.locked()
            lock_b_busy = self._voc_lock_b.locked()
            if not lock_a_busy and lock_b_busy:
                target_gpu = 0
            elif not lock_b_busy and lock_a_busy:
                target_gpu = 1
            else:
                with self._voc_rr_lock:
                    self._voc_rr_index = 1 - self._voc_rr_index
                    target_gpu = self._voc_rr_index

        if target_gpu == 0:
            if self.voc_a is None:
                self.voc_a = self.gen_a.create_vocoder()
            voc, lock = self.voc_a, self._voc_lock_a
        else:
            if self.voc_b is None:
                self.voc_b = self.gen_b.create_vocoder()
            voc, lock = self.voc_b, self._voc_lock_b

        with lock:
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(target_gpu)
                except Exception:
                    pass
            return voc.decode_tiled(tokens, ctx_frames=ctx_frames, step_frames=step_frames)

    def _persistent_generator_loop(
        self,
        worker_name: str,
        gen: GeneratorHandle,
        gpu_id: int,
    ):
        """
        Persistent GPU island worker loop. Continuously admits chunks from self._tts_queue
        under DynamicCapacityCoordinator's active token ceiling and steps all active sessions
        together via continuous batched burst execution.
        """
        active_sessions: Dict[int, Dict[str, Any]] = {}
        current_active_tokens = 0
        pending_item = None
        last_hb_time = time.time()

        def _retire_session(sid: int, error_msg: Optional[str] = None, cancelled: bool = False):
            nonlocal current_active_tokens
            s = active_sessions.pop(sid, None)
            if not s:
                return
            try:
                gen.session_free(sid)
            except Exception:
                pass
            current_active_tokens = max(0, current_active_tokens - s["tokens"])
            self.dynamic_coord.update_active_tokens(gpu_id, current_active_tokens)

            t_end = time.time()
            audio_s = s["total_frames"] * 0.08
            wall_s = t_end - s["t_exec_start"]
            turnaround_s = t_end - s["arr_time"]
            wait_s = s["t_exec_start"] - s["arr_time"]
            ttfa = (s["first_step_time"] - s["arr_time"]) if s["first_step_time"] else turnaround_s
            rtf = wall_s / max(0.01, audio_s)
            stats = {
                "session_id": sid,
                "task_id": s["task"].id,
                "words": s["words"],
                "total_frames": s["total_frames"],
                "audio_s": round(audio_s, 3),
                "queue_wait_s": round(wait_s, 3),
                "compute_wall_s": round(wall_s, 3),
                "turnaround_s": round(turnaround_s, 3),
                "ttfa_s": round(ttfa, 3),
                "rtf": round(rtf, 3),
                "worker": f"GPU {gpu_id} ({worker_name} [{s['model_tag']}])",
                "cancelled": cancelled,
            }
            cb = s["on_complete"]
            if cb:
                try:
                    if cancelled:
                        cb(None, "CANCELLED", stats)
                    elif error_msg:
                        cb(None, error_msg, stats)
                    else:
                        cb(s["all_tokens"], None, stats)
                except Exception as cb_err:
                    print(f"[{worker_name}] Completion callback error for session {sid}: {cb_err}")

        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_id)
            except Exception:
                pass

        while self._running:
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(gpu_id)
                except Exception:
                    pass
            # A. Admit waiting chunks under active_token_budget & max_slots_per_gpu
            while self._running:
                max_active_tokens = self.dynamic_coord.get_admission_ceiling(gpu_id)
                if len(active_sessions) >= self.max_slots_per_gpu:
                    break

                if pending_item is not None:
                    item = pending_item
                else:
                    try:
                        # Block briefly only when completely idle to avoid spinning CPU
                        timeout = 0.02 if not active_sessions else 0.0
                        if timeout > 0:
                            item = self._tts_queue.get(timeout=timeout)
                        else:
                            item = self._tts_queue.get_nowait()
                    except queue.Empty:
                        break

                if item is None:
                    pending_item = None
                    break

                sid, task_item, arr_time, on_complete, cancel_event = item

                # Skip immediately if already cancelled before GPU admission
                if cancel_event is not None and cancel_event.is_set():
                    pending_item = None
                    if on_complete:
                        try:
                            on_complete(None, "CANCELLED", {"session_id": sid, "cancelled": True})
                        except Exception:
                            pass
                    continue

                cfg_scale = float(getattr(task_item, "cfg_scale", 1.0))
                profile = estimate_speech_profile(
                    task_item.text,
                    getattr(task_item, "instruction", None),
                    getattr(task_item, "ref_frames", 0),
                    cfg_scale=cfg_scale,
                    ref_text=getattr(task_item, "ref_text", None),
                )
                needed_tokens = profile["total_needed"]

                # Active token VRAM budget guard
                if (current_active_tokens + needed_tokens > max_active_tokens) and len(active_sessions) > 0:
                    pending_item = item
                    break

                pending_item = None
                max_slot_output = max(64, 2040 - profile["prefix_tokens"])
                if getattr(task_item, "max_steps", 0) > 0:
                    output_cap_frames = min(max_slot_output, task_item.max_steps)
                else:
                    output_cap_frames = min(max_slot_output, profile["conservative_max"])

                use_q4 = (
                    self.enable_q4_burst
                    and self.has_q4_dd
                    and len(active_sessions) >= self.q4_threshold
                )
                model_tag = "Q4" if use_q4 else "Q8"
                t_exec_start = time.time()

                try:
                    cb0, allocated_tokens = gen.session_create_ext(
                        session_id=sid,
                        text=task_item.text,
                        instruction=task_item.instruction,
                        ref_text=task_item.ref_text,
                        ref_codes=task_item.ref_codes,
                        ref_frames=task_item.ref_frames,
                        cfg_scale=cfg_scale,
                        seed=int(task_item.seed),
                        max_new_tokens=output_cap_frames,
                        use_q4=use_q4,
                        temperature=float(getattr(task_item, "temperature", 0.60)),
                        top_p=float(getattr(task_item, "top_p", 0.88)),
                        top_k=int(getattr(task_item, "top_k", 40)),
                        repetition_penalty=float(getattr(task_item, "repetition_penalty", 1.1)),
                        depth_temperature=float(getattr(task_item, "depth_temperature", 0.0)),
                        depth_cfg_codebooks=int(getattr(task_item, "depth_cfg_codebooks", 15)),
                    )
                except Exception as e:
                    if len(active_sessions) > 0:
                        # Transient VRAM pressure: hold until an active stream finishes
                        pending_item = item
                        break
                    else:
                        if on_complete:
                            try:
                                on_complete(None, f"Session prefill failed on GPU {gpu_id}: {e}", {"session_id": sid})
                            except Exception:
                                pass
                        continue

                current_active_tokens += allocated_tokens
                self.dynamic_coord.update_active_tokens(gpu_id, current_active_tokens)

                active_sessions[sid] = {
                    "task": task_item,
                    "model_tag": model_tag,
                    "arr_time": arr_time,
                    "t_exec_start": t_exec_start,
                    "words": profile["words"],
                    "tokens": allocated_tokens,
                    "expected_eos": profile["expected_eos"],
                    "cb0": cb0,
                    "total_frames": 0,
                    "all_tokens": [],
                    "first_step_time": None,
                    "max_steps": output_cap_frames,
                    "on_complete": on_complete,
                    "cancel_event": cancel_event,
                }

            if not active_sessions:
                continue

            # Check for mid-generation cancellations before stepping GPU
            cancelled_sids = [
                sid
                for sid, s in active_sessions.items()
                if s["cancel_event"] is not None and s["cancel_event"].is_set()
            ]
            for sid in cancelled_sids:
                _retire_session(sid, cancelled=True)

            if not active_sessions:
                continue

            # B. Step all active sessions concurrently in one continuous GPU burst
            active_sids = list(active_sessions.keys())
            K = select_burst_steps(active_sessions, max_burst=self.quantum_frames)

            try:
                # Pass seeds=None so each session's mt19937 advances naturally without re-seeding
                next_cb0s, burst_frames = gen.sessions_step_burst(
                    active_sids, burst_steps=K, seeds=None
                )
            except Exception as step_err:
                print(f"[{worker_name}] Burst step error on GPU {gpu_id}: {step_err}")
                for sid in active_sids:
                    _retire_session(sid, error_msg=str(step_err))
                continue

            # C. Collect frames and retire finished streams
            finished_sids = []
            for i, sid in enumerate(active_sids):
                s = active_sessions.get(sid)
                if not s:
                    continue

                s_frames = burst_frames[i]
                final_cb0 = next_cb0s[i]
                stream_hit_eos = False
                valid_count = 0

                for frame16 in s_frames:
                    cb0 = frame16[0]
                    if cb0 < 0 or cb0 == 2051 or cb0 == 4096:
                        stream_hit_eos = True
                        break
                    s["total_frames"] += 1
                    s["all_tokens"].extend(frame16)
                    valid_count += 1
                    if s["total_frames"] >= s["max_steps"]:
                        stream_hit_eos = True
                        break

                if s["first_step_time"] is None and valid_count > 0:
                    s["first_step_time"] = time.time()

                s["cb0"] = final_cb0
                if stream_hit_eos or final_cb0 < 0 or s["total_frames"] >= s["max_steps"]:
                    finished_sids.append(sid)

            for sid in finished_sids:
                _retire_session(sid)

            if time.time() - last_hb_time > 10.0 and active_sessions:
                last_hb_time = time.time()
                frames_list = [s["total_frames"] for s in active_sessions.values()]
                print(
                    f"  [Heartbeat GPU {gpu_id}] Active: {len(active_sessions):2d} streams | "
                    f"Tokens: {current_active_tokens}/{self.dynamic_coord.get_admission_ceiling(gpu_id)} | "
                    f"Frames: min {min(frames_list):3d} / max {max(frames_list):3d}",
                    flush=True,
                )

        # Shutdown cleanup: free any remaining sessions on this GPU
        for sid in list(active_sessions.keys()):
            _retire_session(sid, error_msg="Cluster shutting down")

    def encode_audio(self, pcm_samples: List[float], gpu_id: Optional[int] = None) -> Tuple[List[int], int]:
        """Encodes 24 kHz mono float32 audio samples into 16-codebook tokens at 12.5 Hz (max 10s)."""
        MAX_SAMPLES = 240000
        if len(pcm_samples) > MAX_SAMPLES:
            pcm_samples = pcm_samples[:MAX_SAMPLES]
        
        target_gpu = gpu_id
        if target_gpu is None:
            # Pick GPU 1 if Enhancer is NOT running, else GPU 0
            target_gpu = 1 if not getattr(self, "is_enhancer_running", False) else 0

        gen = self.gen_b if target_gpu == 1 else self.gen_a
        acquired = self.audio_encoder_pool.acquire(blocking=True, timeout=15.0)
        if not acquired:
            raise TimeoutError(f"Audio reference encoder concurrency limit (2 jobs) exceeded on GPU {target_gpu}")
        
        # Request temporary headroom for the ~465 MiB audio encoder compute graph
        self.dynamic_coord.request_headroom(target_gpu, tokens_needed=2000, timeout=10.0)
        try:
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(target_gpu)
                except Exception:
                    pass
            return gen.encode_audio(pcm_samples)
        finally:
            self.dynamic_coord.release_headroom(target_gpu, tokens_needed=2000)
            self.audio_encoder_pool.release()

    def encode_voice_container(self, pcm_samples: List[float], transcript: str, gpu_id: Optional[int] = None) -> bytes:
        codes, frames = self.encode_audio(pcm_samples, gpu_id=gpu_id)
        return serialize_breeze_voice(text=transcript.strip(), codes=codes, frames=frames)

    def convert_voice_task(
        self,
        task: VoiceConversionTask,
        gpu_id: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[List[float]], Optional[List[int]], Dict[str, Any]]:
        profile = estimate_conversion_profile(
            src_frames=task.src_frames,
            ref_frames=task.ref_frames,
            ref_text=task.ref_text,
            src_text=task.src_text,
            cfg_scale=task.cfg_scale,
        )
        needed_tokens = profile["total_needed"]

        if gpu_id is None:
            target_gpu = 0 if self.conv_pool_a.active_count <= self.conv_pool_b.active_count else 1
        else:
            target_gpu = gpu_id

        pool = self.conv_pool_a if target_gpu == 0 else self.conv_pool_b
        gen = self.gen_a if target_gpu == 0 else self.gen_b
        worker_tag = f"CUDA{target_gpu}"

        if not pool.acquire(blocking=True, timeout=timeout):
            raise TimeoutError(f"Conversion concurrency limit ({pool.max_concurrent}) exceeded on {worker_tag}")

        t_start = time.time()
        try:
            audio, tokens = gen.convert_voice(
                src_codes=task.src_codes,
                src_T=task.src_frames,
                ref_audio=task.ref_audio,
                ref_codes=task.ref_codes,
                ref_frames=task.ref_frames,
                ref_text=task.ref_text,
                src_text=task.src_text,
                cfg_scale=task.cfg_scale,
                keep_acoustic=task.keep_acoustic,
                feed_source=task.feed_source,
                seed=task.seed,
                return_pcm=not task.client_has_vocoder,
            )
            wall_s = time.time() - t_start
            audio_s = task.src_frames * (1920.0 / 24000.0)
            stats = {
                "id": task.id,
                "worker": worker_tag,
                "src_frames": task.src_frames,
                "audio_s": audio_s,
                "compute_wall_s": wall_s,
                "rtf": wall_s / max(0.01, audio_s),
                "tokens_reserved": needed_tokens,
                "vram_estimate_mib": profile["vram_estimate_mib"],
            }
            return audio, tokens, stats
        finally:
            pool.release()

    def run_workload(
        self,
        tasks: List[Any],
        out_dir: str = "audio_out",
        arrival_delays: Optional[List[float]] = None,
        on_progress: Optional[Callable[[ClusterResult], None]] = None,
        on_pcm_chunk: Optional[Callable[[int, np.ndarray, bool], None]] = None,
        quantum_frames: int = 16,
        max_slots_per_gpu: Optional[int] = None,
        vocoder_chunk_size: int = 32,
        return_tokens: bool = False,
        max_active_tokens_per_gpu: Optional[int] = None,
    ) -> List[ClusterResult]:
        """
        Backward-compatible batch runner built directly on top of the persistent dual-GPU
        cluster (submit_tts_chunk + decode_tokens_tiled). Used by CLI and benchmark scripts.
        """
        os.makedirs(out_dir, exist_ok=True)
        conversion_tasks = [t for t in tasks if isinstance(t, VoiceConversionTask)]
        tts_tasks = [t for t in tasks if not isinstance(t, VoiceConversionTask)]

        completed_results: List[ClusterResult] = []
        results_lock = threading.Lock()
        done_event = threading.Event()
        remaining = [len(tts_tasks) + len(conversion_tasks)]

        if remaining[0] == 0:
            return []

        def _mark_done(res: ClusterResult):
            with results_lock:
                completed_results.append(res)
                if on_progress:
                    on_progress(res)
                remaining[0] -= 1
                if remaining[0] <= 0:
                    done_event.set()

        for idx, task in enumerate(tts_tasks):
            delay = arrival_delays[idx] if arrival_delays and idx < len(arrival_delays) else 0.0
            if delay > 0:
                time.sleep(delay)

            def _make_cb(t: UserTask):
                def _cb(tokens: Optional[List[int]], err: Optional[str], stats: Dict[str, Any]):
                    if err or tokens is None:
                        _mark_done(
                            ClusterResult(
                                id=t.id,
                                words=stats.get("words", len(t.text.split())),
                                audio_s=0.0,
                                queue_wait_s=stats.get("queue_wait_s", 0.0),
                                compute_wall_s=stats.get("compute_wall_s", 0.0),
                                turnaround_s=stats.get("turnaround_s", 0.0),
                                ttfa_s=0.0,
                                rtf=0.0,
                                worker=stats.get("worker", "unknown"),
                                wav_path="",
                                tokens=None,
                                success=False,
                                error_msg=err or "TTS generation failed",
                            )
                        )
                        return

                    wav_path = ""
                    audio_s = stats.get("audio_s", (len(tokens) // 16) * 0.08)
                    if not return_tokens and not getattr(t, "client_has_vocoder", False):
                        try:
                            pcm = self.decode_tokens_tiled(tokens)
                            audio_s = len(pcm) / float(self.sample_rate)
                            if on_pcm_chunk:
                                on_pcm_chunk(t.id, pcm, True)
                            wav_path = os.path.join(out_dir, f"dual_gen_user_{t.id:02d}.wav")
                            pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
                            with wave.open(wav_path, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(self.sample_rate)
                                wf.writeframes(pcm_int16.tobytes())
                        except Exception as voc_err:
                            _mark_done(
                                ClusterResult(
                                    id=t.id,
                                    words=stats.get("words", len(t.text.split())),
                                    audio_s=0.0,
                                    queue_wait_s=stats.get("queue_wait_s", 0.0),
                                    compute_wall_s=stats.get("compute_wall_s", 0.0),
                                    turnaround_s=stats.get("turnaround_s", 0.0),
                                    ttfa_s=0.0,
                                    rtf=0.0,
                                    worker=stats.get("worker", "unknown"),
                                    wav_path="",
                                    tokens=tokens,
                                    success=False,
                                    error_msg=f"Tiled vocoder error: {voc_err}",
                                )
                            )
                            return

                    _mark_done(
                        ClusterResult(
                            id=t.id,
                            words=stats.get("words", len(t.text.split())),
                            audio_s=round(audio_s, 2),
                            queue_wait_s=round(stats.get("queue_wait_s", 0.0), 2),
                            compute_wall_s=round(stats.get("compute_wall_s", 0.0), 2),
                            turnaround_s=round(stats.get("turnaround_s", 0.0), 2),
                            ttfa_s=round(stats.get("ttfa_s", 0.0), 3),
                            rtf=round(stats.get("rtf", 0.0), 3),
                            worker=stats.get("worker", "DualGPU"),
                            wav_path=wav_path,
                            tokens=tokens if (return_tokens or getattr(t, "client_has_vocoder", False)) else None,
                            success=True,
                        )
                    )

                return _cb

            self.submit_tts_chunk(task, _make_cb(task))

        for c_task in conversion_tasks:
            def _run_conv(ct: VoiceConversionTask):
                t0 = time.time()
                try:
                    audio, tokens, stats = self.convert_voice_task(ct)
                    wav_path = ""
                    if audio:
                        wav_path = os.path.join(out_dir, f"conversion_user_{ct.id:04d}_{stats['worker']}.wav")
                        pcm_16 = (np.clip(np.array(audio), -1.0, 1.0) * 32767.0).astype(np.int16)
                        with wave.open(wav_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(self.sample_rate)
                            wf.writeframes(pcm_16.tobytes())
                    _mark_done(
                        ClusterResult(
                            id=ct.id,
                            words=int(math.ceil(stats["audio_s"] * (140.0 / 60.0))),
                            audio_s=stats["audio_s"],
                            queue_wait_s=0.0,
                            compute_wall_s=stats["compute_wall_s"],
                            turnaround_s=time.time() - t0,
                            ttfa_s=stats["compute_wall_s"],
                            rtf=stats["rtf"],
                            worker=stats["worker"],
                            wav_path=wav_path,
                            tokens=tokens,
                            success=True,
                        )
                    )
                except Exception as conv_err:
                    _mark_done(
                        ClusterResult(
                            id=ct.id,
                            words=0,
                            audio_s=0.0,
                            queue_wait_s=0.0,
                            compute_wall_s=time.time() - t0,
                            turnaround_s=time.time() - t0,
                            ttfa_s=0.0,
                            rtf=0.0,
                            worker="conversion",
                            wav_path="",
                            success=False,
                            error_msg=str(conv_err),
                        )
                    )

            threading.Thread(target=_run_conv, args=(c_task,), daemon=True).start()

        done_event.wait()
        completed_results.sort(key=lambda x: x.id)
        return completed_results

    def close(self):
        self._running = False
        # Wake both persistent worker threads
        for _ in range(2):
            try:
                self._tts_queue.put_nowait(None)
            except Exception:
                pass
        for t in (getattr(self, "_worker_a", None), getattr(self, "_worker_b", None)):
            if t and t.is_alive():
                t.join(timeout=2.0)

        for attr in ("voc_a", "voc_b", "gen_a", "gen_b"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def __del__(self):
        self.close()
