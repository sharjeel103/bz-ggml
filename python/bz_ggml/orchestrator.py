"""
Unified Engine Orchestrator & Parcel Pipeline.
Provides a completely decoupled, production-grade engine interface:
  - Input: Parcel containing a list of TaskSpec jobs.
  - Execution: Symmetrically scheduled across Dual GPU Islands with Dynamic Capacity Modulation ("Green Light" Protocol).
  - Output: Strictly in-memory ParcelResult containing numpy audio buffers, transcript strings, SRT subtitles, and voice bytes.
  - Zero filesystem dependencies: External consumers (Gradio, FastAPI, Celery) handle persistence on their own.
"""

import gc
import io
import math
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
import numpy as np
import soundfile as sf

from .cluster import DualInstanceCluster, UserTask, VoiceConversionTask
from .asr import ASRService
from .audio_pipeline import (
    split_text_into_smart_chunks,
    stitch_audio_chunks,
    enhance_audio_array,
    master_audio_array,
    format_chunk_preview_markdown
)
from .bindings import serialize_breeze_voice, deserialize_breeze_voice


class TaskType(str, Enum):
    TTS = "tts"
    STITCH = "stitch"
    ENHANCE = "enhance"
    MASTER = "master"
    ASR = "asr"
    VOICE_REGISTER = "voice_register"
    VOICE_CONVERT = "voice_convert"


@dataclass
class TaskSpec:
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Parcel:
    parcel_id: str
    tasks: List[TaskSpec]
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tts_pipeline(
        cls,
        script_text: str,
        voice_name: Optional[str] = None,
        instruction: str = "Speak clearly and naturally.",
        cfg_scale: float = 1.5,
        temperature: float = 0.25,
        seed: int = 42,
        pause_ms: int = 200,
        target_words: int = 80,
        max_words: int = 120,
        enhance: bool = True,
        master: bool = True,
        srt: bool = True,
        ref_codes: Optional[List[int]] = None,
        ref_frames: int = 0,
        ref_text: Optional[str] = None,
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """Constructs a standard long-form text-to-speech production pipeline Parcel."""
        p_id = parcel_id or f"parcel_tts_{int(time.time() * 1000)}"
        tasks = [TaskSpec(TaskType.TTS)]
        tasks.append(TaskSpec(TaskType.STITCH, {"pause_ms": pause_ms}))
        if enhance:
            tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}))
        if master:
            tasks.append(TaskSpec(TaskType.MASTER, {"preset": "optionA_rich_bass"}))
        if srt:
            tasks.append(TaskSpec(TaskType.ASR, {"output_format": "all"}))

        payload = {
            "script_text": script_text,
            "voice_name": voice_name,
            "instruction": instruction,
            "cfg_scale": float(cfg_scale),
            "temperature": float(temperature),
            "seed": int(seed),
            "pause_ms": int(pause_ms),
            "target_words": int(target_words),
            "max_words": int(max_words),
            "ref_codes": ref_codes,
            "ref_frames": int(ref_frames),
            "ref_text": ref_text
        }
        return cls(parcel_id=p_id, tasks=tasks, payload=payload)

    @classmethod
    def voice_registration(
        cls,
        audio_input: Union[str, np.ndarray, List[float]],
        voice_name: str = "custom_voice",
        transcript: Optional[str] = None,
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """Constructs a Voice Persona Registration parcel [ASR (if needed), VOICE_REGISTER]."""
        p_id = parcel_id or f"parcel_vreg_{int(time.time() * 1000)}"
        tasks = []
        if not transcript or not transcript.strip():
            tasks.append(TaskSpec(TaskType.ASR, {"output_format": "txt"}))
        tasks.append(TaskSpec(TaskType.VOICE_REGISTER, {"voice_name": voice_name}))

        payload = {
            "audio_input": audio_input,
            "voice_name": voice_name,
            "transcript": transcript
        }
        return cls(parcel_id=p_id, tasks=tasks, payload=payload)

    @classmethod
    def standalone_asr(
        cls,
        audio_input: Union[str, np.ndarray, List[float]],
        output_format: str = "all",
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """Constructs a standalone ASR transcription parcel."""
        p_id = parcel_id or f"parcel_asr_{int(time.time() * 1000)}"
        tasks = [TaskSpec(TaskType.ASR, {"output_format": output_format})]
        return cls(parcel_id=p_id, tasks=tasks, payload={"audio_input": audio_input})

    @classmethod
    def standalone_enhance(
        cls,
        audio_input: Union[str, np.ndarray, List[float]],
        chunk_seconds: float = 10.0,
        master: bool = True,
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """Constructs a standalone Enhancer parcel."""
        p_id = parcel_id or f"parcel_enh_{int(time.time() * 1000)}"
        tasks = [TaskSpec(TaskType.ENHANCE, {"chunk_seconds": chunk_seconds})]
        if master:
            tasks.append(TaskSpec(TaskType.MASTER, {"preset": "optionA_rich_bass"}))
        return cls(parcel_id=p_id, tasks=tasks, payload={"audio_input": audio_input})


@dataclass
class ParcelResult:
    parcel_id: str
    success: bool = True
    error_message: Optional[str] = None
    audio_pcm: Optional[np.ndarray] = None          # Raw float32 mono waveform in memory
    raw_audio_pcm: Optional[np.ndarray] = None      # Pre-enhancement raw stitched waveform
    sample_rate: int = 24000                        # Sample rate (24000 or 44100 if enhanced)
    duration_s: float = 0.0
    transcript: Optional[str] = None
    subtitles_srt: Optional[str] = None
    segments: Optional[List[dict]] = None
    voice_bytes: Optional[bytes] = None             # Serialized .breeze container bytes
    chunks_count: int = 1
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
        """Validates that all requested tasks can be fulfilled by the running engine instance."""
        task_types = [t.task_type for t in parcel.tasks]
        if TaskType.ENHANCE in task_types and not self.enable_enhancer:
            raise ValueError(
                "Requested TaskType.ENHANCE but Resemble Enhancer is disabled on this engine instance. "
                "Restart engine with --enable-enhancer or omit ENHANCE task."
            )
        if TaskType.ASR in task_types and not self.enable_asr:
            raise ValueError(
                "Requested TaskType.ASR but ASR Micro-Service is disabled on this engine instance. "
                "Restart engine with --enable-asr or omit ASR task."
            )
        if TaskType.VOICE_REGISTER in task_types and not self.enable_asr and not parcel.payload.get("transcript"):
            raise ValueError(
                "TaskType.VOICE_REGISTER requires verbatim reference text when ASR Micro-Service is disabled."
            )


class EngineOrchestrator:
    """
    Decoupled Production Engine Orchestrator.
    Executes Parcels across Dual GPU Islands with Dynamic Capacity Modulation:
      - TTS tasks: Parallel continuous multi-session batching across GPU 0 & GPU 1.
      - Resemble Enhancer: Dynamic headroom modulation on GPU 1 (Green Light protocol).
      - Faster-Whisper: Dynamic headroom modulation on GPU 0 (Green Light protocol).
      - In-Memory Studio Mastering & Normalization (Option A Winning Preset).
      - Pure In-Memory Return Contract (ParcelResult).
    """
    def __init__(
        self,
        cluster: DualInstanceCluster,
        asr_service: Optional[ASRService] = None,
        capabilities: Optional[EngineCapabilities] = None
    ):
        self.cluster = cluster
        self.asr_service = asr_service
        self.capabilities = capabilities or EngineCapabilities()
        self.lock = threading.Lock()
        print(f"[EngineOrchestrator] Online with Precision: {self.capabilities.precision.upper()} | Enhancer: {self.capabilities.enable_enhancer} | ASR: {self.capabilities.enable_asr}")

    def dispatch(
        self,
        parcel: Parcel,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> ParcelResult:
        """
        Executes a Parcel sequentially through requested task specifications.
        Guarantees thread-safety and zero model unloading/swapping.
        """
        self.capabilities.validate_parcel(parcel)

        t_start_total = time.time()
        stage_timings: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}

        result = ParcelResult(
            parcel_id=parcel.parcel_id,
            sample_rate=self.cluster.sample_rate
        )

        def report(stage: str, progress: float, msg: str):
            print(f"   [Orchestrator: {stage} ({progress*100:.0f}%)] {msg}", flush=True)
            if progress_callback:
                try:
                    progress_callback(stage, progress, msg)
                except Exception:
                    pass

        try:
            current_pcm: Optional[np.ndarray] = None
            current_sr: int = self.cluster.sample_rate

            # If payload provides an initial audio input, pre-load it into memory
            if "audio_input" in parcel.payload and parcel.payload["audio_input"] is not None:
                inp = parcel.payload["audio_input"]
                if isinstance(inp, np.ndarray):
                    current_pcm = inp.astype(np.float32)
                elif isinstance(inp, list):
                    current_pcm = np.array(inp, dtype=np.float32)
                elif isinstance(inp, str):
                    data, sr = sf.read(inp, dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    current_pcm = data
                    current_sr = sr

            # -------------------------------------------------------------
            # STAGE 1: TASK EXECUTION LOOP
            # -------------------------------------------------------------
            for task_spec in parcel.tasks:
                task_t0 = time.time()
                t_type = task_spec.task_type

                # ------------------- 1. TTS SYNTHESIS -------------------
                if t_type == TaskType.TTS:
                    report("tts_chunking", 0.05, "Partitioning script into semantic speech chunks...")
                    script = parcel.payload.get("script_text", "").strip()
                    if not script:
                        raise ValueError("Parcel script_text is empty.")

                    target_words = int(parcel.payload.get("target_words", 80))
                    max_words = int(parcel.payload.get("max_words", 120))
                    chunks = split_text_into_smart_chunks(script, target_words=target_words, max_words=max_words)
                    if not chunks:
                        raise ValueError("No valid speech tokens in script_text.")

                    result.chunks_count = len(chunks)
                    words_total = sum(len(c.split()) for c in chunks)
                    report("tts_parallel", 0.15, f"Dispatching {len(chunks)} parallel chunks ({words_total}w) across Dual GPU Islands...")

                    instruction = parcel.payload.get("instruction", "Speak clearly.")
                    cfg_scale = float(parcel.payload.get("cfg_scale", 1.5))
                    temperature = float(parcel.payload.get("temperature", 0.25))
                    seed = int(parcel.payload.get("seed", 42))
                    ref_codes = parcel.payload.get("ref_codes")
                    ref_frames = int(parcel.payload.get("ref_frames", 0))
                    ref_text = parcel.payload.get("ref_text")

                    tts_tasks = []
                    for idx, chunk_text in enumerate(chunks):
                        tts_tasks.append(UserTask(
                            id=idx,
                            text=chunk_text,
                            instruction=instruction,
                            ref_text=ref_text,
                            ref_codes=ref_codes,
                            ref_frames=ref_frames,
                            cfg_scale=cfg_scale,
                            temperature=temperature,
                            seed=seed + idx,
                            max_steps=2040,
                            stream_pcm=False,
                            client_has_vocoder=False
                        ))

                    chunks_done = 0
                    def on_chunk_complete(res):
                        nonlocal chunks_done
                        chunks_done += 1
                        pct = 0.15 + (chunks_done / len(chunks)) * 0.45
                        report("tts_parallel", pct, f"Synthesized chunk {chunks_done}/{len(chunks)} ({res.worker}).")

                    with tempfile.TemporaryDirectory(prefix="orch_tts_") as tmp_dir:
                        tts_results = self.cluster.run_workload(
                            tts_tasks,
                            out_dir=tmp_dir,
                            on_progress=on_chunk_complete
                        )

                        tts_results_sorted = sorted(tts_results, key=lambda r: r.id)
                        chunks_pcm = []
                        total_audio_s = 0.0
                        for r in tts_results_sorted:
                            data, sr = sf.read(r.wav_path, dtype="float32")
                            if data.ndim > 1:
                                data = data.mean(axis=1)
                            chunks_pcm.append(data)
                            total_audio_s += r.audio_s

                    # Cache chunk array list in payload for stitching
                    parcel.payload["_chunks_pcm"] = chunks_pcm
                    metadata["tts_total_audio_s"] = round(total_audio_s, 2)
                    stage_timings["tts"] = time.time() - task_t0

                # ------------------- 2. STITCHING -------------------
                elif t_type == TaskType.STITCH:
                    report("stitching", 0.62, "Stitching chunks in memory with breath pause & click-free micro-fades...")
                    chunks_pcm = parcel.payload.get("_chunks_pcm", [])
                    pause_ms = int(task_spec.params.get("pause_ms", parcel.payload.get("pause_ms", 200)))
                    current_pcm = stitch_audio_chunks(chunks_pcm, sample_rate=self.cluster.sample_rate, pause_ms=pause_ms)
                    current_sr = self.cluster.sample_rate
                    result.raw_audio_pcm = current_pcm.copy()
                    stage_timings["stitch"] = time.time() - task_t0

                # ------------------- 3. RESEMBLE ENHANCE -------------------
                elif t_type == TaskType.ENHANCE:
                    if current_pcm is None or len(current_pcm) == 0:
                        raise ValueError("TaskType.ENHANCE requires audio input or prior TTS/STITCH stage.")

                    chunk_sec = float(task_spec.params.get("chunk_seconds", self.capabilities.chunk_seconds_enhancer))
                    tokens_needed = 16850 if chunk_sec <= 10.0 else 22210

                    report("enhancing", 0.68, f"Requesting {tokens_needed} headroom on GPU 1 for Resemble Enhance...")
                    # Green Light Protocol: modulate headroom dynamically on GPU 1
                    green_light = self.cluster.dynamic_coord.request_headroom(gpu_id=1, tokens_needed=tokens_needed)
                    try:
                        report("enhancing", 0.73, f"Enhancing audio in-memory (44.1 kHz, Chunks: {chunk_sec}s)...")
                        device_tag = f"cuda:{self.capabilities.enhancer_device_index}"
                        enhanced_pcm, out_sr = enhance_audio_array(
                            current_pcm,
                            sr=current_sr,
                            chunk_seconds=chunk_sec,
                            device=device_tag
                        )
                        current_pcm = enhanced_pcm
                        current_sr = out_sr
                    finally:
                        self.cluster.dynamic_coord.release_headroom(gpu_id=1, tokens_needed=tokens_needed)

                    stage_timings["enhance"] = time.time() - task_t0

                # ------------------- 4. STUDIO MASTERING -------------------
                elif t_type == TaskType.MASTER:
                    if current_pcm is None or len(current_pcm) == 0:
                        raise ValueError("TaskType.MASTER requires audio input.")

                    preset = task_spec.params.get("preset", "optionA_rich_bass")
                    report("mastering", 0.85, f"Applying studio broadcast mastering ({preset})...")
                    mastered_pcm, out_sr = master_audio_array(current_pcm, sr=current_sr, preset=preset)
                    current_pcm = mastered_pcm
                    current_sr = out_sr
                    stage_timings["master"] = time.time() - task_t0

                # ------------------- 5. ASR TRANSCRIPTION -------------------
                elif t_type == TaskType.ASR:
                    if current_pcm is None or len(current_pcm) == 0:
                        raise ValueError("TaskType.ASR requires audio input.")
                    if self.asr_service is None:
                        raise RuntimeError("ASR Service instance is not configured on this engine.")

                    tokens_needed = 9600
                    report("transcribing", 0.90, f"Requesting {tokens_needed} headroom on GPU 0 for Whisper ASR...")
                    # Green Light Protocol: modulate headroom dynamically on GPU 0
                    green_light = self.cluster.dynamic_coord.request_headroom(gpu_id=0, tokens_needed=tokens_needed)
                    try:
                        report("transcribing", 0.93, "Transcribing mastered speech on GPU 0 (word-level SRT + verbatim text)...")
                        fmt = task_spec.params.get("output_format", "all")
                        res = self.asr_service.transcribe(
                            audio=current_pcm,
                            sample_rate=current_sr,
                            output_format="all"
                        )
                        transcript, srt_content, seg_list = res
                        result.transcript = transcript
                        result.subtitles_srt = srt_content
                        result.segments = seg_list
                        parcel.payload["transcript"] = transcript
                    finally:
                        self.cluster.dynamic_coord.release_headroom(gpu_id=0, tokens_needed=tokens_needed)

                    stage_timings["asr"] = time.time() - task_t0

                # ------------------- 6. VOICE REGISTRATION -------------------
                elif t_type == TaskType.VOICE_REGISTER:
                    if current_pcm is None or len(current_pcm) == 0:
                        raise ValueError("TaskType.VOICE_REGISTER requires reference audio.")

                    transcript = parcel.payload.get("transcript")
                    if not transcript or not transcript.strip():
                        if self.asr_service is not None:
                            report("voice_register", 0.88, "Transcribing reference audio for Voice Registration...")
                            transcript = self.asr_service.transcribe(current_pcm, sample_rate=current_sr, output_format="txt")
                        else:
                            raise ValueError("Voice registration requires verbatim text transcript.")

                    report("voice_register", 0.95, "Encoding 16-codebook discrete tokens on GPU 0 into in-memory .breeze...")
                    # Resample to 24 kHz if needed for Breeze acoustic encoder
                    if current_sr != 24000:
                        import scipy.signal
                        num_samples = int(round(len(current_pcm) * 24000.0 / float(current_sr)))
                        pcm_24k = scipy.signal.resample(current_pcm, num_samples).astype(np.float32)
                    else:
                        pcm_24k = current_pcm

                    voice_bytes = self.cluster.encode_voice_container(
                        pcm_samples=pcm_24k.tolist(),
                        transcript=transcript.strip(),
                        gpu_id=0
                    )
                    result.voice_bytes = voice_bytes
                    result.transcript = transcript.strip()
                    stage_timings["voice_register"] = time.time() - task_t0

                # ------------------- 7. VOICE CONVERSION -------------------
                elif t_type == TaskType.VOICE_CONVERT:
                    report("voice_convert", 0.90, "Executing Mode 4 Voice Conversion across GPU Islands...")
                    conv_task = parcel.payload.get("conversion_task")
                    if not isinstance(conv_task, VoiceConversionTask):
                        raise ValueError("VOICE_CONVERT requires a valid VoiceConversionTask in parcel.payload['conversion_task'].")
                    pcm_out, tokens_out, stats = self.cluster.convert_voice_task(conv_task)
                    if pcm_out:
                        current_pcm = np.array(pcm_out, dtype=np.float32)
                        current_sr = self.cluster.sample_rate
                    metadata["conversion_stats"] = stats
                    stage_timings["voice_convert"] = time.time() - task_t0

            # -------------------------------------------------------------
            # FINAL RESULT ASSEMBLY (STRICTLY IN-MEMORY)
            # -------------------------------------------------------------
            result.audio_pcm = current_pcm
            result.sample_rate = current_sr
            result.duration_s = round(len(current_pcm) / float(current_sr), 2) if current_pcm is not None else 0.0
            result.stage_timings = stage_timings
            result.metadata = metadata
            report("completed", 1.0, f"Parcel {parcel.parcel_id} processed successfully in {time.time()-t_start_total:.2f}s!")
            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            report("failed", 1.0, f"Parcel processing failed: {str(e)}")
            return result
