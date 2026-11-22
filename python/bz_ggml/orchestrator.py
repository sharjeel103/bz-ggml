"""
Unified Engine Orchestrator, Decoupled Job Queues & Non-Blocking State Machine.
Production non-streaming dual-T4 pipeline:
  - Anchor-First 1 + (N-1) Parallel TTS Subsystem:
      * <= 200 words (and <= 2,000 estimated tokens): single chunk -> tiled vocode.
      * > 200 words: Chunk 0 (25-50 word opening anchor) establishes live voice/tone/energy
        with full CFG -> Chunks 1..N-1 fan out in parallel across GPU 0 & GPU 1 anchored to
        Chunk 0's tokens -> concatenated token stream decoded once via bounded tiled vocoder
        (ctx=80, step=240, <= ~1.2 GB VRAM).
  - Dedicated Stage Queues: persistent DualInstanceCluster TTS queue, Vocode (conc=2),
    Enhance (8, conc=1 on GPU 1), ASR (20, conc=1 on GPU 0), Voice Reg (15, conc=1 on GPU 0),
    Master (20, conc=4 pure in-memory FFmpeg CPU), Upload (100, conc=8 async I/O).
  - Zero temporary WAV files on disk; full in-memory PCM/token hand-offs and real cancellation.
"""

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
import urllib.request

from .cluster import DualInstanceCluster, UserTask, estimate_speech_profile
from .asr import ASRService
from .audio_pipeline import (
    split_text_into_smart_chunks,
    split_text_with_anchor,
    enhance_audio_array,
    master_audio_array,
    load_any_audio,
    encode_pcm_to_mp3_bytes,
    encode_pcm_to_wav_bytes,
)


class TaskType(str, Enum):
    TTS = "tts"
    STITCH = "stitch"
    ENHANCE = "enhance"
    MASTER = "master"
    ASR = "asr"
    VOICE_REGISTER = "voice_register"
    VOICE_CONVERT = "voice_convert"
    UPLOAD = "upload"


@dataclass
class TaskSpec:
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Parcel:
    parcel_id: str
    tasks: List[TaskSpec]
    payload: Dict[str, Any] = field(default_factory=dict)
    return_artifacts: Optional[List[str]] = None
    upload_targets: Optional[Dict[str, str]] = None

    @classmethod
    def tts_pipeline(
        cls,
        script_text: str,
        voice_name: Optional[str] = None,
        instruction: str = "Speak clearly and naturally.",
        cfg_scale: float = 1.5,
        follower_cfg_scale: Optional[float] = None,
        temperature: float = 0.60,
        top_p: float = 0.88,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        depth_temperature: float = 0.0,
        depth_cfg_codebooks: int = 15,
        seed: int = 42,
        pause_ms: int = 200,
        anchor_words: int = 35,
        target_words: int = 110,
        max_words: int = 140,
        enhance: bool = True,
        master: bool = True,
        srt: bool = True,
        ref_codes: Optional[List[int]] = None,
        ref_frames: int = 0,
        ref_text: Optional[str] = None,
        return_artifacts: Optional[List[str]] = None,
        upload_targets: Optional[Dict[str, str]] = None,
        parcel_id: Optional[str] = None,
    ) -> "Parcel":
        """Constructs a standard long-form non-streaming TTS production pipeline Parcel."""
        p_id = parcel_id or f"parcel_tts_{int(time.time() * 1000)}"
        tasks = [TaskSpec(TaskType.TTS)]
        if enhance:
            tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}))
        if master:
            tasks.append(TaskSpec(TaskType.MASTER, {"preset": "optionA_rich_bass"}))
        if srt:
            tasks.append(TaskSpec(TaskType.ASR, {"output_format": "all"}))
        if upload_targets:
            tasks.append(TaskSpec(TaskType.UPLOAD))

        payload = {
            "script_text": script_text,
            "voice_name": voice_name,
            "instruction": instruction,
            "cfg_scale": float(cfg_scale),
            "follower_cfg_scale": float(follower_cfg_scale) if follower_cfg_scale is not None else None,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "depth_temperature": float(depth_temperature),
            "depth_cfg_codebooks": int(depth_cfg_codebooks),
            "seed": int(seed),
            "pause_ms": int(pause_ms),
            "anchor_words": int(anchor_words),
            "target_words": int(target_words),
            "max_words": int(max_words),
            "ref_codes": ref_codes,
            "ref_frames": int(ref_frames),
            "ref_text": ref_text,
            "upload_targets": upload_targets,
        }
        return cls(
            parcel_id=p_id,
            tasks=tasks,
            payload=payload,
            return_artifacts=return_artifacts,
            upload_targets=upload_targets,
        )

    @classmethod
    def voice_registration(
        cls,
        voice_name: str,
        audio_input: Any,
        transcript: Optional[str] = None,
        enhance: bool = True,
        max_duration_s: float = 10.0,
        return_artifacts: Optional[List[str]] = None,
        upload_targets: Optional[Dict[str, str]] = None,
        parcel_id: Optional[str] = None,
    ) -> "Parcel":
        """Constructs a Voice Persona Registration pipeline Parcel with strict <= 10s auto-trimming."""
        p_id = parcel_id or f"parcel_vreg_{int(time.time() * 1000)}"
        tasks = []
        if enhance:
            tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}))
        tasks.append(TaskSpec(TaskType.ASR, {"output_format": "txt"}))
        tasks.append(TaskSpec(TaskType.VOICE_REGISTER))
        if upload_targets:
            tasks.append(TaskSpec(TaskType.UPLOAD))

        payload = {
            "voice_name": voice_name,
            "audio_input": audio_input,
            "transcript": transcript,
            "max_duration_s": float(max_duration_s),
            "upload_targets": upload_targets,
        }
        return cls(
            parcel_id=p_id,
            tasks=tasks,
            payload=payload,
            return_artifacts=return_artifacts,
            upload_targets=upload_targets,
        )


@dataclass
class ParcelResult:
    parcel_id: str
    success: bool = True
    error_message: Optional[str] = None
    duration_s: float = 0.0
    chunks_count: int = 1
    sample_rate: int = 24000
    audio_pcm: Optional[np.ndarray] = None
    raw_audio_pcm: Optional[np.ndarray] = None
    transcript: Optional[str] = None
    subtitles_srt: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    voice_bytes: Optional[bytes] = None
    upload_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifact_bytes: Dict[str, bytes] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineCapabilities:
    """Reflects the static configuration and enabled micro-services of the engine instance."""
    precision: str = "int8"
    enable_q4_surge: bool = False
    vocoder_mode: str = "on"
    enable_enhancer: bool = True
    enable_asr: bool = True
    chunk_seconds_enhancer: float = 10.0
    asr_model_size: str = "turbo"
    asr_device_index: int = 0
    enhancer_device_index: int = 1

    def validate_parcel(self, parcel: Parcel) -> None:
        task_types = [t.task_type for t in parcel.tasks]
        if TaskType.ENHANCE in task_types and not self.enable_enhancer:
            raise ValueError(
                "Requested TaskType.ENHANCE but Resemble Enhancer is disabled on this engine instance."
            )
        if TaskType.ASR in task_types and not self.enable_asr:
            raise ValueError(
                "Requested TaskType.ASR but ASR Micro-Service is disabled on this engine instance."
            )
        if TaskType.VOICE_REGISTER in task_types and not self.enable_asr and not parcel.payload.get("transcript"):
            raise ValueError(
                "TaskType.VOICE_REGISTER requires verbatim reference text when ASR Micro-Service is disabled."
            )
        if TaskType.UPLOAD in task_types:
            upload_targets = parcel.payload.get("upload_targets") or parcel.upload_targets
            if not upload_targets or not isinstance(upload_targets, dict):
                raise ValueError("TaskType.UPLOAD requires a valid dictionary of upload_targets in parcel payload.")
            valid_keys = {"audio_mp3", "audio_wav", "subtitles_srt", "transcript_txt", "voice_breeze", "raw_audio_wav"}
            for k, url in upload_targets.items():
                if k not in valid_keys:
                    raise ValueError(f"Invalid upload target '{k}'. Must be one of {valid_keys}.")
                if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
                    raise ValueError(f"Upload target URL for '{k}' must be a valid HTTP/HTTPS URL.")
                if k in ("subtitles_srt", "transcript_txt") and TaskType.ASR not in task_types and not parcel.payload.get("transcript"):
                    raise ValueError(f"Upload target '{k}' requested but ASR task is missing from parcel.")
                if k == "voice_breeze" and TaskType.VOICE_REGISTER not in task_types:
                    raise ValueError(f"Upload target '{k}' requested but VOICE_REGISTER task is missing from parcel.")


@dataclass
class StageJob:
    job_id: str
    request_id: str
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)
    input_pcm: Optional[np.ndarray] = None
    input_sr: int = 24000
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobDoneEvent:
    request_id: str
    task_type: TaskType
    success: bool = True
    error_message: Optional[str] = None
    output_pcm: Optional[np.ndarray] = None
    output_sr: int = 24000
    artifacts: Dict[str, Any] = field(default_factory=dict)
    stage_timing_s: float = 0.0


class RequestTTSAssembly:
    """
    Per-request token-level assembly state machine for Anchor-First 1 + (N-1) parallel TTS.
    Collects 16-codebook token arrays out-of-order in memory without touching disk.
    """

    def __init__(
        self,
        request_id: str,
        chunks_text: List[str],
        sample_rate: int = 24000,
        started_at: Optional[float] = None,
    ):
        self.request_id = request_id
        self.chunks_text = chunks_text
        self.total_chunks = len(chunks_text)
        self.sample_rate = sample_rate
        self.started_at = started_at or time.time()
        self.completed_count = 0
        self.failed = False
        self.chunks_tokens: List[Optional[List[int]]] = [None] * self.total_chunks
        self.lock = threading.Lock()

    def record_chunk_tokens(self, chunk_index: int, tokens: List[int]) -> Optional[List[int]]:
        """
        Records a completed chunk's 16-codebook tokens.
        Returns the concatenated full token stream when all chunks have arrived, else None.
        """
        with self.lock:
            if self.failed:
                return None
            if self.chunks_tokens[chunk_index] is None:
                self.chunks_tokens[chunk_index] = tokens
                self.completed_count += 1

            if self.completed_count == self.total_chunks:
                merged: List[int] = []
                for c_toks in self.chunks_tokens:
                    if c_toks:
                        merged.extend(c_toks)
                return merged
        return None

    def mark_failed(self) -> bool:
        """Marks assembly failed; returns True only on the first failure to avoid duplicate events."""
        with self.lock:
            if self.failed:
                return False
            self.failed = True
            return True


@dataclass
class RequestContext:
    request_id: str
    parcel: Parcel
    remaining_tasks: List[TaskSpec]
    current_task: Optional[TaskSpec] = None
    current_pcm: Optional[np.ndarray] = None
    current_sr: int = 24000
    raw_audio_pcm: Optional[np.ndarray] = None
    transcript: Optional[str] = None
    subtitles_srt: Optional[str] = None
    voice_bytes: Optional[bytes] = None
    upload_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifact_bytes: Dict[str, bytes] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_stages: int = 1
    completed_stages: int = 0
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    status: str = "QUEUED"
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    stage_started_at: float = field(default_factory=time.time)
    estimated_duration_s: float = 1.0
    completion_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[ParcelResult] = None
    progress_callback: Optional[Callable[[str, float, str], None]] = None


class EngineOrchestrator:
    """
    Decoupled Production Engine Orchestrator.
    Routes parcels through dedicated stage queues while delegating parallel TTS execution
    directly to the persistent DualInstanceCluster.
    """

    MAX_QUEUE_TTS = 150
    MAX_QUEUE_ENHANCE = 8
    MAX_QUEUE_ASR = 20
    MAX_QUEUE_VOICE_REG = 15
    MAX_QUEUE_MASTER = 20
    MAX_QUEUE_UPLOAD = 100
    SINGLE_CHUNK_MAX_WORDS = 200
    SINGLE_CHUNK_MAX_TOKENS = 2000

    def __init__(
        self,
        cluster: DualInstanceCluster,
        asr_service: Optional[ASRService] = None,
        capabilities: Optional[EngineCapabilities] = None,
    ):
        self.cluster = cluster
        self.asr_service = asr_service
        self.capabilities = capabilities or EngineCapabilities()

        # Dedicated Stage Queues
        self.vocode_queue: queue.Queue = queue.Queue()
        self.enhance_queue: queue.Queue[StageJob] = queue.Queue(maxsize=self.MAX_QUEUE_ENHANCE)
        self.asr_queue: queue.Queue[StageJob] = queue.Queue(maxsize=self.MAX_QUEUE_ASR)
        self.voice_reg_queue: queue.Queue[StageJob] = queue.Queue(maxsize=self.MAX_QUEUE_VOICE_REG)
        self.master_queue: queue.Queue[StageJob] = queue.Queue(maxsize=self.MAX_QUEUE_MASTER)
        self.upload_queue: queue.Queue[StageJob] = queue.Queue(maxsize=self.MAX_QUEUE_UPLOAD)

        # Non-blocking Completion Inbox
        self.completion_queue: queue.Queue[JobDoneEvent] = queue.Queue()

        # State Registries
        self.active_requests: Dict[str, RequestContext] = {}
        self.tts_assemblies: Dict[str, RequestTTSAssembly] = {}
        self._pending_tts_chunks = 0
        self.state_lock = threading.RLock()
        self.running = True

        self._threads: List[threading.Thread] = []
        self._start_worker_pools()

        print(
            f"[EngineOrchestrator] Online (Non-Streaming Anchor-First Parallel) | "
            f"Precision: {self.capabilities.precision.upper()} | "
            f"Enhancer: {self.capabilities.enable_enhancer} | ASR: {self.capabilities.enable_asr}"
        )

    def shutdown(self):
        self.running = False

    def validate_parcel(self, parcel: Parcel) -> None:
        self.capabilities.validate_parcel(parcel)

    def get_queue_depths(self) -> Dict[str, int]:
        with self.state_lock:
            pending_tts = self._pending_tts_chunks
        return {
            "tts": pending_tts,
            "vocode": self.vocode_queue.qsize(),
            "enhance": self.enhance_queue.qsize(),
            "asr": self.asr_queue.qsize(),
            "voice_reg": self.voice_reg_queue.qsize(),
            "master": self.master_queue.qsize(),
            "upload": self.upload_queue.qsize(),
        }

    def estimate_parcel_duration(self, parcel: Parcel) -> float:
        dur = 0.0
        task_types = [t.task_type for t in parcel.tasks]
        if TaskType.TTS in task_types:
            script = parcel.payload.get("script_text", "")
            words = len(script.split()) if script else 40
            dur += max(1.5, (words / 100.0) * 3.5)
        if TaskType.ENHANCE in task_types:
            dur += 6.5
        if TaskType.ASR in task_types:
            dur += 0.25
        if TaskType.MASTER in task_types:
            dur += 0.35
        if TaskType.VOICE_REGISTER in task_types:
            dur += 1.5
        if TaskType.UPLOAD in task_types:
            dur += 0.5
        return round(max(0.5, dur), 2)

    def evaluate_admission(self, parcel: Parcel) -> Tuple[bool, str, float, float, int]:
        """Multi-stage capacity admission check."""
        task_types = [t.task_type for t in parcel.tasks]

        with self.state_lock:
            if TaskType.TTS in task_types:
                script = parcel.payload.get("script_text", "")
                words = len(script.split()) if script else 40
                chunks_needed = 1 if words <= self.SINGLE_CHUNK_MAX_WORDS else max(2, math.ceil(words / 110))
                if self._pending_tts_chunks + chunks_needed > self.MAX_QUEUE_TTS:
                    return (
                        False,
                        f"REJECTED_STAGE_OVERFLOW: Stage 'tts' capacity ({self.MAX_QUEUE_TTS} chunks) is saturated.",
                        0.0,
                        0.0,
                        self._pending_tts_chunks,
                    )

            if TaskType.ENHANCE in task_types and self.enhance_queue.qsize() >= self.MAX_QUEUE_ENHANCE:
                wait_s = round((self.enhance_queue.qsize() + 1) * 6.5, 1)
                return (
                    False,
                    f"REJECTED_STAGE_OVERFLOW: Stage 'enhance' queue ({self.MAX_QUEUE_ENHANCE}) is saturated. ETA: {wait_s}s.",
                    0.0,
                    0.0,
                    self.enhance_queue.qsize(),
                )

            if TaskType.ASR in task_types and self.asr_queue.qsize() >= self.MAX_QUEUE_ASR:
                wait_s = round((self.asr_queue.qsize() + 1) * 0.25, 1)
                return (
                    False,
                    f"REJECTED_STAGE_OVERFLOW: Stage 'asr' queue ({self.MAX_QUEUE_ASR}) is saturated. ETA: {wait_s}s.",
                    0.0,
                    0.0,
                    self.asr_queue.qsize(),
                )

            if TaskType.VOICE_REGISTER in task_types and self.voice_reg_queue.qsize() >= self.MAX_QUEUE_VOICE_REG:
                wait_s = round((self.voice_reg_queue.qsize() + 1) * 1.5, 1)
                return (
                    False,
                    f"REJECTED_STAGE_OVERFLOW: Stage 'voice_register' queue ({self.MAX_QUEUE_VOICE_REG}) is saturated. ETA: {wait_s}s.",
                    0.0,
                    0.0,
                    self.voice_reg_queue.qsize(),
                )

            if TaskType.UPLOAD in task_types and self.upload_queue.qsize() >= self.MAX_QUEUE_UPLOAD:
                return (
                    False,
                    f"REJECTED_STAGE_OVERFLOW: Stage 'upload' queue ({self.MAX_QUEUE_UPLOAD}) is saturated.",
                    0.0,
                    0.0,
                    self.upload_queue.qsize(),
                )

            duration_s = self.estimate_parcel_duration(parcel)
            drain_wait_s = 0.0
            if TaskType.ENHANCE in task_types:
                drain_wait_s = max(drain_wait_s, self.enhance_queue.qsize() * 6.5)
            if TaskType.ASR in task_types:
                drain_wait_s = max(drain_wait_s, self.asr_queue.qsize() * 0.25)
            if TaskType.TTS in task_types:
                drain_wait_s = max(drain_wait_s, (self._pending_tts_chunks / 4.0) * 3.5)

            total_eta_s = round(drain_wait_s + duration_s, 2)
            pos = max(1, len(self.active_requests) + 1)
            return (True, "ADMITTED", duration_s, total_eta_s, pos)

    def cancel_request(self, request_id: str) -> bool:
        """Immediately cancels an active request, aborting any queued or active GPU chunks."""
        with self.state_lock:
            req = self.active_requests.get(request_id)
            if not req:
                return False
            req.cancel_event.set()
            req.status = "CANCELLED"
            req.error_message = "Request cancelled by caller."
            assembly = self.tts_assemblies.pop(request_id, None)
            if assembly:
                assembly.mark_failed()
            self._finalize_request(req)
            return True

    def dispatch(
        self,
        parcel: Parcel,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> ParcelResult:
        return self.submit_sync(parcel, progress_callback=progress_callback)

    def submit_sync(
        self,
        parcel: Parcel,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        timeout: float = 600.0,
        skip_admission: bool = False,
    ) -> ParcelResult:
        self.validate_parcel(parcel)

        if not skip_admission:
            accepted, reason, dur_s, eta_s, pos = self.evaluate_admission(parcel)
            if not accepted:
                return ParcelResult(
                    parcel_id=parcel.parcel_id,
                    success=False,
                    error_message=reason,
                )

        req = self._create_request_context(parcel, progress_callback=progress_callback)

        if "audio_input" in parcel.payload and parcel.payload["audio_input"] is not None:
            max_dur = parcel.payload.get("max_duration_s")
            task_types = [t.task_type for t in parcel.tasks]
            if max_dur is None and TaskType.VOICE_REGISTER in task_types:
                max_dur = 10.0
            pcm, sr = load_any_audio(parcel.payload["audio_input"], target_sr=None, max_duration_s=max_dur)
            req.current_pcm = pcm
            req.current_sr = sr
            if TaskType.STITCH not in task_types and TaskType.TTS not in task_types:
                req.raw_audio_pcm = pcm.copy()

        self._enqueue_next_task(req)

        finished = req.completion_event.wait(timeout=timeout)
        if not finished:
            self.cancel_request(req.request_id)
            return ParcelResult(
                parcel_id=parcel.parcel_id,
                success=False,
                error_message=f"Execution timed out after {timeout}s.",
            )

        return req.result or ParcelResult(
            parcel_id=parcel.parcel_id,
            success=req.status == "COMPLETED",
            error_message=req.error_message,
        )

    def _create_request_context(
        self,
        parcel: Parcel,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> RequestContext:
        req_id = parcel.parcel_id
        tasks_copy = list(parcel.tasks)
        req = RequestContext(
            request_id=req_id,
            parcel=parcel,
            remaining_tasks=tasks_copy,
            total_stages=max(1, len(tasks_copy)),
            progress_callback=progress_callback,
            current_sr=getattr(self.cluster, "sample_rate", 24000),
        )
        with self.state_lock:
            self.active_requests[req_id] = req
        return req

    def _enqueue_next_task(self, req: RequestContext):
        with self.state_lock:
            if req.cancel_event.is_set():
                return
            if not req.remaining_tasks:
                self._finalize_request(req)
                return

            next_task = req.remaining_tasks.pop(0)
            req.current_task = next_task
            t_type = next_task.task_type
            req.status = "RUNNING"
            req.stage_started_at = time.time()

            pct = req.completed_stages / float(req.total_stages)
            req.overall_progress = round(pct, 3)
            self._report_progress(req, str(t_type.value), 0.05, f"Starting stage '{t_type.value}'...")

            job = StageJob(
                job_id=f"job_{req.request_id}_{t_type.value}",
                request_id=req.request_id,
                task_type=t_type,
                params=next_task.params,
                input_pcm=req.current_pcm,
                input_sr=req.current_sr,
                extra=req.parcel.payload,
            )

            if t_type == TaskType.TTS:
                self._handle_tts_dispatch(req, job)
            elif t_type == TaskType.ENHANCE:
                self.enhance_queue.put(job)
            elif t_type == TaskType.ASR:
                self.asr_queue.put(job)
            elif t_type == TaskType.VOICE_REGISTER:
                self.voice_reg_queue.put(job)
            elif t_type == TaskType.MASTER:
                self.master_queue.put(job)
            elif t_type == TaskType.UPLOAD:
                self.upload_queue.put(job)
            elif t_type == TaskType.STITCH:
                self._handle_standalone_stitch(req, job)
            else:
                self.completion_queue.put(JobDoneEvent(request_id=req.request_id, task_type=t_type, success=True))

    def _prepare_tts_chunks(self, script: str, payload: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Applies the 200-word AND 2,000-token single-chunk guard.
        Scripts exceeding either limit are split using Anchor-First (Chunk 0 = 25-50w anchor,
        Chunks 1..N-1 = target_words..max_words) honoring payload parameters.
        """
        if payload is None:
            payload = {}
        words = script.split()
        ref_frames = int(payload.get("ref_frames", 0) or 0)
        ref_text = payload.get("ref_text")
        instruction = payload.get("instruction", "Speak clearly and naturally.")
        cfg_scale = float(payload.get("cfg_scale", 1.5))

        profile = estimate_speech_profile(
            script,
            instruction=instruction,
            ref_frames=ref_frames,
            cfg_scale=cfg_scale,
            ref_text=ref_text,
        )
        single_branch_est = profile["prefix_tokens"] + profile["conservative_max"]
        if len(words) <= self.SINGLE_CHUNK_MAX_WORDS and single_branch_est <= self.SINGLE_CHUNK_MAX_TOKENS:
            return [script]

        anchor_words = int(payload.get("anchor_words", 35))
        target_words = min(140, max(40, int(payload.get("target_words", 110))))
        max_words = min(170, max(target_words + 10, int(payload.get("max_words", 140))))
        return split_text_with_anchor(
            script,
            anchor_target_words=anchor_words,
            anchor_max_words=max(anchor_words + 15, 50),
            target_words=target_words,
            max_words=max_words,
        )

    def _handle_tts_dispatch(self, req: RequestContext, job: StageJob):
        """
        Event-driven Anchor-First 1 + (N-1) Parallel TTS Dispatcher.
          - Single chunk: submits directly to persistent DualInstanceCluster.
          - Multi-chunk: submits Chunk 0 first; when Chunk 0 finishes, uses Chunk 0's
            (text, tokens) as the live acoustic anchor for parallel Chunks 1..N-1 across
            both T4 GPUs, keeping CFG active for consistent expressive speech.
        """
        script = (job.extra.get("script_text") or "").strip()
        if not script:
            self.completion_queue.put(
                JobDoneEvent(
                    request_id=req.request_id,
                    task_type=TaskType.TTS,
                    success=False,
                    error_message="Parcel script_text is empty.",
                )
            )
            return

        chunks = self._prepare_tts_chunks(script, job.extra)
        total_chunks = len(chunks)
        req.metadata["chunks_count"] = total_chunks

        assembly = RequestTTSAssembly(
            request_id=req.request_id,
            chunks_text=chunks,
            sample_rate=req.current_sr,
            started_at=time.time(),
        )
        with self.state_lock:
            self.tts_assemblies[req.request_id] = assembly
            self._pending_tts_chunks += total_chunks

        instruction = job.extra.get("instruction", "Speak clearly and naturally.")
        cfg_scale = float(job.extra.get("cfg_scale", 1.5))
        follower_cfg = job.extra.get("follower_cfg_scale")
        follower_cfg_scale = float(follower_cfg) if follower_cfg is not None else cfg_scale
        temperature = float(job.extra.get("temperature", 0.60))
        top_p = float(job.extra.get("top_p", 0.88))
        top_k = int(job.extra.get("top_k", 40))
        repetition_penalty = float(job.extra.get("repetition_penalty", 1.1))
        depth_temperature = float(job.extra.get("depth_temperature", 0.0))
        depth_cfg_codebooks = int(job.extra.get("depth_cfg_codebooks", 15))
        seed = int(job.extra.get("seed", 42))
        ref_codes = job.extra.get("ref_codes")
        ref_frames = int(job.extra.get("ref_frames", 0) or 0)
        ref_text = job.extra.get("ref_text")

        def _dec_pending(count: int = 1):
            with self.state_lock:
                self._pending_tts_chunks = max(0, self._pending_tts_chunks - count)

        def _fail_assembly(err_msg: str, remaining_unsubmitted: int = 0):
            if remaining_unsubmitted > 0:
                _dec_pending(remaining_unsubmitted)
            if assembly.mark_failed():
                with self.state_lock:
                    self.tts_assemblies.pop(req.request_id, None)
                if not req.cancel_event.is_set():
                    self.completion_queue.put(
                        JobDoneEvent(
                            request_id=req.request_id,
                            task_type=TaskType.TTS,
                            success=False,
                            error_message=err_msg,
                            stage_timing_s=time.time() - assembly.started_at,
                        )
                    )

        def _on_chunk_done(chunk_idx: int, tokens: Optional[List[int]], err: Optional[str], stats: Dict[str, Any]):
            _dec_pending(1)
            if req.cancel_event.is_set() or err == "CANCELLED":
                assembly.mark_failed()
                with self.state_lock:
                    self.tts_assemblies.pop(req.request_id, None)
                return

            if err or tokens is None:
                rem = (total_chunks - 1) if (chunk_idx == 0 and total_chunks > 1) else 0
                _fail_assembly(err or f"TTS chunk {chunk_idx} failed.", remaining_unsubmitted=rem)
                return

            merged_tokens = assembly.record_chunk_tokens(chunk_idx, tokens)
            pct = 0.10 + 0.75 * (assembly.completed_count / float(total_chunks))
            self._report_progress(
                req,
                "tts",
                pct,
                f"Generated chunk {assembly.completed_count}/{total_chunks}...",
            )

            # Anchor-First Fan-Out: Chunk 0 just completed -> launch Chunks 1..N-1 in parallel!
            if chunk_idx == 0 and total_chunks > 1 and not assembly.failed:
                anchor_frames = len(tokens) // 16
                # Cap anchor reference at ~120 frames (~9.6s) so Breeze never skips follower text
                if anchor_frames > 120:
                    anchor_codes = tokens[: 120 * 16]
                    anchor_frames = 120
                else:
                    anchor_codes = tokens
                anchor_text = chunks[0]

                self._report_progress(
                    req,
                    "tts",
                    0.20,
                    f"Anchor locked ({anchor_frames * 0.08:.1f}s); fanning out {total_chunks - 1} parallel chunks across GPU 0 & GPU 1...",
                )
                for f_idx in range(1, total_chunks):
                    f_task = UserTask(
                        id=f_idx,
                        text=chunks[f_idx],
                        instruction=instruction,
                        ref_text=anchor_text,
                        ref_codes=anchor_codes,
                        ref_frames=anchor_frames,
                        cfg_scale=follower_cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        depth_temperature=depth_temperature,
                        depth_cfg_codebooks=depth_cfg_codebooks,
                        seed=seed + f_idx,
                        max_steps=2040,
                        stream_pcm=False,
                        client_has_vocoder=True,  # Return raw tokens for continuous tiled vocoding
                    )
                    self.cluster.submit_tts_chunk(
                        f_task,
                        on_complete=lambda tok, e, st, i=f_idx: _on_chunk_done(i, tok, e, st),
                        cancel_event=req.cancel_event,
                    )

            # All chunks complete -> hand off merged token stream to tiled vocoder worker
            if merged_tokens is not None:
                with self.state_lock:
                    self.tts_assemblies.pop(req.request_id, None)
                self._report_progress(req, "tts", 0.88, "Running continuous tiled vocoder...")
                self.vocode_queue.put((req.request_id, merged_tokens, assembly.started_at, req.cancel_event))

        self._report_progress(
            req,
            "tts",
            0.10,
            f"Submitting {'anchor chunk 0 of ' + str(total_chunks) if total_chunks > 1 else 'single chunk'} to dual-T4 cluster...",
        )

        # Submit Chunk 0 (or the sole chunk) to persistent cluster
        chunk0_task = UserTask(
            id=0,
            text=chunks[0],
            instruction=instruction,
            ref_text=ref_text,
            ref_codes=ref_codes,
            ref_frames=ref_frames,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            depth_temperature=depth_temperature,
            depth_cfg_codebooks=depth_cfg_codebooks,
            seed=seed,
            max_steps=2040,
            stream_pcm=False,
            client_has_vocoder=True,  # Always return tokens for continuous tiled vocoding
        )
        self.cluster.submit_tts_chunk(
            chunk0_task,
            on_complete=lambda tok, e, st: _on_chunk_done(0, tok, e, st),
            cancel_event=req.cancel_event,
        )

    def _handle_standalone_stitch(self, req: RequestContext, job: StageJob):
        if req.current_pcm is not None:
            self.completion_queue.put(
                JobDoneEvent(
                    request_id=req.request_id,
                    task_type=TaskType.STITCH,
                    success=True,
                    output_pcm=req.current_pcm,
                    output_sr=req.current_sr,
                )
            )
        else:
            self.completion_queue.put(
                JobDoneEvent(
                    request_id=req.request_id,
                    task_type=TaskType.STITCH,
                    success=False,
                    error_message="STITCH requested without prior audio buffers.",
                )
            )

    def _report_progress(self, req: RequestContext, stage: str, pct: float, msg: str):
        req.stage_progress = round(pct, 3)
        overall = (req.completed_stages + pct) / float(req.total_stages)
        req.overall_progress = round(min(0.99, max(0.01, overall)), 3)
        if req.progress_callback:
            try:
                req.progress_callback(stage, pct, msg)
            except Exception:
                pass

    def _orchestration_loop(self):
        while self.running:
            try:
                event = self.completion_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.state_lock:
                req = self.active_requests.get(event.request_id)
                if not req or req.cancel_event.is_set():
                    continue

                if not event.success:
                    req.status = "FAILED"
                    req.error_message = event.error_message or f"Task '{event.task_type.value}' failed."
                    self._finalize_request(req)
                    continue

                req.completed_stages += 1
                req.stage_timings[event.task_type.value] = round(event.stage_timing_s, 3)

                if event.output_pcm is not None:
                    req.current_pcm = event.output_pcm
                    req.current_sr = event.output_sr
                    if req.raw_audio_pcm is None and event.task_type in (TaskType.TTS, TaskType.STITCH):
                        req.raw_audio_pcm = event.output_pcm.copy()

                for k, v in event.artifacts.items():
                    req.artifacts[k] = v
                    if k == "transcript":
                        req.transcript = v
                    elif k == "subtitles_srt":
                        req.subtitles_srt = v
                    elif k == "voice_bytes":
                        req.voice_bytes = v
                    elif k == "upload_results":
                        req.upload_results.update(v)
                    elif k == "fallback_artifact_bytes":
                        req.artifact_bytes.update(v)

                self._report_progress(req, event.task_type.value, 1.0, f"Completed stage '{event.task_type.value}'.")
                self._enqueue_next_task(req)

    def _finalize_request(self, req: RequestContext):
        with self.state_lock:
            if req.status not in ("FAILED", "CANCELLED"):
                req.status = "COMPLETED"
            req.overall_progress = 1.0

            result = ParcelResult(
                parcel_id=req.request_id,
                success=(req.status == "COMPLETED"),
                error_message=req.error_message,
                duration_s=round(len(req.current_pcm) / float(req.current_sr), 2) if req.current_pcm is not None else 0.0,
                chunks_count=req.metadata.get("chunks_count", 1),
                sample_rate=req.current_sr,
                audio_pcm=req.current_pcm,
                raw_audio_pcm=req.raw_audio_pcm,
                transcript=req.transcript,
                subtitles_srt=req.subtitles_srt,
                voice_bytes=req.voice_bytes,
                upload_results=req.upload_results,
                artifact_bytes=req.artifact_bytes,
                stage_timings=req.stage_timings,
                metadata=req.metadata,
            )

            if req.status == "COMPLETED" and req.parcel.return_artifacts:
                for art_name in req.parcel.return_artifacts:
                    if art_name not in result.artifact_bytes:
                        if art_name == "audio_mp3" and req.current_pcm is not None:
                            result.artifact_bytes["audio_mp3"] = encode_pcm_to_mp3_bytes(req.current_pcm, req.current_sr)
                        elif art_name == "audio_wav" and req.current_pcm is not None:
                            result.artifact_bytes["audio_wav"] = encode_pcm_to_wav_bytes(req.current_pcm, req.current_sr)
                        elif art_name == "subtitles_srt" and req.subtitles_srt:
                            result.artifact_bytes["subtitles_srt"] = req.subtitles_srt.encode("utf-8")
                        elif art_name == "transcript_txt" and req.transcript:
                            result.artifact_bytes["transcript_txt"] = req.transcript.encode("utf-8")
                        elif art_name == "voice_breeze" and req.voice_bytes:
                            result.artifact_bytes["voice_breeze"] = req.voice_bytes
                        elif art_name == "raw_audio_wav" and req.raw_audio_pcm is not None:
                            result.artifact_bytes["raw_audio_wav"] = encode_pcm_to_wav_bytes(req.raw_audio_pcm, 24000)

            req.result = result
            req.completion_event.set()
            if req.status == "COMPLETED":
                self._report_progress(req, "completed", 1.0, f"Request {req.request_id} finished successfully!")

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self.state_lock:
            req_snapshots = []
            stale_ids = []
            for r_id, req in self.active_requests.items():
                if req.status in ("COMPLETED", "FAILED", "CANCELLED") and (now - req.created_at > 60.0):
                    stale_ids.append(r_id)
                    continue
                curr_stage = req.current_task.task_type.value if req.current_task else "idle"
                elapsed = round(now - req.created_at, 2)
                eta_rem = max(0.0, round(req.estimated_duration_s * (1.0 - req.overall_progress), 2))
                req_snapshots.append({
                    "request_id": r_id,
                    "status": req.status,
                    "current_stage": curr_stage,
                    "stage_progress": req.stage_progress,
                    "overall_progress": req.overall_progress,
                    "elapsed_s": elapsed,
                    "eta_s": eta_rem,
                })

            for sid in stale_ids:
                del self.active_requests[sid]
                self.tts_assemblies.pop(sid, None)

            return {
                "queue_depths": self.get_queue_depths(),
                "active_requests": req_snapshots,
            }

    def _start_worker_pools(self):
        t_orch = threading.Thread(target=self._orchestration_loop, daemon=True, name="OrchEventLoop")
        t_orch.start()
        self._threads.append(t_orch)

        # 2 Tiled Vocoder workers (1 per GPU) so vocoding never blocks TTS generator callbacks
        for i in range(2):
            t_voc = threading.Thread(target=self._vocode_worker_loop, args=(i,), daemon=True, name=f"VocodeWorker-{i}")
            t_voc.start()
            self._threads.append(t_voc)

        t_enh = threading.Thread(target=self._enhance_worker_loop, daemon=True, name="EnhanceWorker")
        t_enh.start()
        self._threads.append(t_enh)

        t_asr = threading.Thread(target=self._asr_worker_loop, daemon=True, name="ASRWorker")
        t_asr.start()
        self._threads.append(t_asr)

        t_vreg = threading.Thread(target=self._voice_reg_worker_loop, daemon=True, name="VoiceRegWorker")
        t_vreg.start()
        self._threads.append(t_vreg)

        for i in range(4):
            t_mas = threading.Thread(target=self._master_worker_loop, daemon=True, name=f"MasterWorker-{i}")
            t_mas.start()
            self._threads.append(t_mas)

        for i in range(8):
            t_upl = threading.Thread(target=self._upload_worker_loop, daemon=True, name=f"UploadWorker-{i}")
            t_upl.start()
            self._threads.append(t_upl)

    def _vocode_worker_loop(self, worker_idx: int):
        """Decodes merged 16-codebook token streams using strictly VRAM-bounded tiled vocoding."""
        while self.running:
            try:
                req_id, merged_tokens, tts_started_at, cancel_event = self.vocode_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cancel_event is not None and cancel_event.is_set():
                continue

            try:
                pcm = self.cluster.decode_tokens_tiled(
                    merged_tokens,
                    ctx_frames=80,
                    step_frames=240,
                    gpu_id=worker_idx % 2,
                )
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=req_id,
                        task_type=TaskType.TTS,
                        success=True,
                        output_pcm=pcm,
                        output_sr=getattr(self.cluster, "sample_rate", 24000),
                        stage_timing_s=time.time() - tts_started_at,
                    )
                )
            except Exception as e:
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=req_id,
                        task_type=TaskType.TTS,
                        success=False,
                        error_message=f"Tiled vocoder failed: {e}",
                        stage_timing_s=time.time() - tts_started_at,
                    )
                )

    def _enhance_worker_loop(self):
        while self.running:
            try:
                job = self.enhance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                pcm = job.input_pcm
                if pcm is None or len(pcm) == 0:
                    raise ValueError("ENHANCE stage requires audio input.")

                chunk_sec = float(job.params.get("chunk_seconds", self.capabilities.chunk_seconds_enhancer))
                tokens_needed = 16850 if chunk_sec <= 10.0 else 22210

                if hasattr(self.cluster, "dynamic_coord"):
                    self.cluster.dynamic_coord.request_headroom(gpu_id=1, tokens_needed=tokens_needed)
                try:
                    device_tag = f"cuda:{self.capabilities.enhancer_device_index}"
                    enhanced_pcm, out_sr = enhance_audio_array(
                        pcm,
                        sr=job.input_sr,
                        chunk_seconds=chunk_sec,
                        device=device_tag,
                    )
                finally:
                    if hasattr(self.cluster, "dynamic_coord"):
                        self.cluster.dynamic_coord.release_headroom(gpu_id=1, tokens_needed=tokens_needed)

                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.ENHANCE,
                        success=True,
                        output_pcm=enhanced_pcm,
                        output_sr=out_sr,
                        stage_timing_s=time.time() - t0,
                    )
                )
            except Exception as e:
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.ENHANCE,
                        success=False,
                        error_message=str(e),
                        stage_timing_s=time.time() - t0,
                    )
                )

    def _asr_worker_loop(self):
        while self.running:
            try:
                job = self.asr_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                pcm = job.input_pcm
                if pcm is None or len(pcm) == 0:
                    raise ValueError("ASR stage requires audio input.")
                if self.asr_service is None:
                    raise RuntimeError("ASR Service instance is not configured on this engine.")

                tokens_needed = 9600
                if hasattr(self.cluster, "dynamic_coord"):
                    self.cluster.dynamic_coord.request_headroom(gpu_id=0, tokens_needed=tokens_needed)
                try:
                    transcript, srt_content, seg_list = self.asr_service.transcribe(
                        audio=pcm,
                        sample_rate=job.input_sr,
                        output_format="all",
                    )
                finally:
                    if hasattr(self.cluster, "dynamic_coord"):
                        self.cluster.dynamic_coord.release_headroom(gpu_id=0, tokens_needed=tokens_needed)

                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.ASR,
                        success=True,
                        output_pcm=pcm,
                        output_sr=job.input_sr,
                        artifacts={
                            "transcript": transcript,
                            "subtitles_srt": srt_content,
                            "segments": seg_list,
                        },
                        stage_timing_s=time.time() - t0,
                    )
                )
            except Exception as e:
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.ASR,
                        success=False,
                        error_message=str(e),
                        stage_timing_s=time.time() - t0,
                    )
                )

    def _voice_reg_worker_loop(self):
        while self.running:
            try:
                job = self.voice_reg_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                pcm = job.input_pcm
                if pcm is None or len(pcm) == 0:
                    raise ValueError("VOICE_REGISTER stage requires reference audio.")

                transcript = job.extra.get("transcript")
                if not transcript or not transcript.strip():
                    if self.asr_service is not None:
                        transcript = self.asr_service.transcribe(pcm, sample_rate=job.input_sr, output_format="txt")
                    else:
                        raise ValueError("Voice registration requires verbatim text transcript.")

                if job.input_sr != 24000:
                    import scipy.signal
                    gcd = math.gcd(int(job.input_sr), 24000)
                    pcm_24k = scipy.signal.resample_poly(pcm, 24000 // gcd, int(job.input_sr) // gcd).astype(np.float32)
                else:
                    pcm_24k = pcm

                voice_bytes = self.cluster.encode_voice_container(
                    pcm_samples=pcm_24k.tolist(),
                    transcript=transcript.strip(),
                    gpu_id=0,
                )

                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.VOICE_REGISTER,
                        success=True,
                        output_pcm=pcm,
                        output_sr=job.input_sr,
                        artifacts={
                            "voice_bytes": voice_bytes,
                            "transcript": transcript.strip(),
                        },
                        stage_timing_s=time.time() - t0,
                    )
                )
            except Exception as e:
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.VOICE_REGISTER,
                        success=False,
                        error_message=str(e),
                        stage_timing_s=time.time() - t0,
                    )
                )

    def _master_worker_loop(self):
        while self.running:
            try:
                job = self.master_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                pcm = job.input_pcm
                if pcm is None or len(pcm) == 0:
                    raise ValueError("MASTER stage requires audio input.")

                preset = job.params.get("preset", "optionA_rich_bass")
                mastered_pcm, out_sr = master_audio_array(pcm, sr=job.input_sr, preset=preset)

                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.MASTER,
                        success=True,
                        output_pcm=mastered_pcm,
                        output_sr=out_sr,
                        stage_timing_s=time.time() - t0,
                    )
                )
            except Exception as e:
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.MASTER,
                        success=False,
                        error_message=str(e),
                        stage_timing_s=time.time() - t0,
                    )
                )

    def _upload_worker_loop(self):
        while self.running:
            try:
                job = self.upload_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                upload_targets = job.extra.get("upload_targets") or {}
                if not upload_targets:
                    self.completion_queue.put(
                        JobDoneEvent(
                            request_id=job.request_id,
                            task_type=TaskType.UPLOAD,
                            success=True,
                            stage_timing_s=time.time() - t0,
                        )
                    )
                    continue

                upload_results = {}
                fallback_bytes = {}
                any_failed = False
                current_pcm = job.input_pcm
                current_sr = job.input_sr

                with self.state_lock:
                    req = self.active_requests.get(job.request_id)
                    subs = req.subtitles_srt if req else None
                    txt = req.transcript if req else None
                    v_bytes = req.voice_bytes if req else None
                    raw_pcm = req.raw_audio_pcm if req else None

                def get_bytes(name: str) -> Tuple[Optional[bytes], str]:
                    if name == "audio_mp3" and current_pcm is not None:
                        return encode_pcm_to_mp3_bytes(current_pcm, current_sr), "audio/mpeg"
                    elif name == "audio_wav" and current_pcm is not None:
                        return encode_pcm_to_wav_bytes(current_pcm, current_sr), "audio/wav"
                    elif name == "subtitles_srt" and subs:
                        return subs.encode("utf-8"), "application/x-subrip"
                    elif name == "transcript_txt" and txt:
                        return txt.encode("utf-8"), "text/plain; charset=utf-8"
                    elif name == "voice_breeze" and v_bytes:
                        return v_bytes, "application/octet-stream"
                    elif name == "raw_audio_wav" and raw_pcm is not None:
                        return encode_pcm_to_wav_bytes(raw_pcm, 24000), "audio/wav"
                    return None, "application/octet-stream"

                for art_name, target_url in upload_targets.items():
                    art_bytes, mime = get_bytes(art_name)
                    if art_bytes is None:
                        upload_results[art_name] = {"success": False, "error": f"Artifact '{art_name}' not available."}
                        any_failed = True
                        continue

                    try:
                        req_http = urllib.request.Request(
                            target_url,
                            data=art_bytes,
                            headers={"Content-Type": mime, "Content-Length": str(len(art_bytes))},
                            method="PUT",
                        )
                        with urllib.request.urlopen(req_http, timeout=30.0) as resp:
                            if resp.status in (200, 201, 204):
                                upload_results[art_name] = {"success": True, "status": resp.status}
                            else:
                                upload_results[art_name] = {"success": False, "status": resp.status}
                                any_failed = True
                    except Exception as up_err:
                        upload_results[art_name] = {"success": False, "error": str(up_err)}
                        any_failed = True

                if any_failed:
                    for art_name in upload_targets.keys():
                        b, _ = get_bytes(art_name)
                        if b:
                            fallback_bytes[art_name] = b

                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.UPLOAD,
                        success=True,
                        output_pcm=current_pcm,
                        output_sr=current_sr,
                        artifacts={
                            "upload_results": upload_results,
                            "fallback_artifact_bytes": fallback_bytes,
                        },
                        stage_timing_s=time.time() - t0,
                    )
                )
            except Exception as e:
                self.completion_queue.put(
                    JobDoneEvent(
                        request_id=job.request_id,
                        task_type=TaskType.UPLOAD,
                        success=False,
                        error_message=str(e),
                        stage_timing_s=time.time() - t0,
                    )
                )
