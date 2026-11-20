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
    format_chunk_preview_markdown,
    load_any_audio,
    encode_pcm_to_mp3_bytes,
    encode_pcm_to_wav_bytes
)
import urllib.request
from .bindings import serialize_breeze_voice, deserialize_breeze_voice


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
        return_artifacts: Optional[List[str]] = None,
        upload_targets: Optional[Dict[str, str]] = None,
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
        if upload_targets:
            tasks.append(TaskSpec(TaskType.UPLOAD))

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
            "ref_text": ref_text,
            "upload_targets": upload_targets
        }
        return cls(
            parcel_id=p_id,
            tasks=tasks,
            payload=payload,
            return_artifacts=return_artifacts,
            upload_targets=upload_targets
        )

    @classmethod
    def voice_registration(
        cls,
        audio_input: Union[str, bytes, np.ndarray, List[float]],
        voice_name: str = "custom_voice",
        enhance: bool = True,
        max_duration_s: float = 10.0,
        transcript: Optional[str] = None,
        return_artifacts: Optional[List[str]] = None,
        upload_targets: Optional[Dict[str, str]] = None,
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """
        Constructs a modular Voice Persona Registration parcel:
          [Clip <= 10s] -> [Optional Enhance on GPU 1] -> [Whisper Turbo ASR on GPU 0] -> [Breeze Acoustic Encode on GPU 0] -> [Optional Upload / Return]
        Guarantees:
          - Zero manual human transcription required (transcribed automatically).
          - Strict max_duration_s ceiling enforced at ingress so Enhancer, ASR, and Encoder all process identical audio slice.
        """
        p_id = parcel_id or f"parcel_vreg_{int(time.time() * 1000)}"
        tasks = []
        if enhance:
            tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}))
        tasks.append(TaskSpec(TaskType.ASR, {"output_format": "txt"}))
        tasks.append(TaskSpec(TaskType.VOICE_REGISTER, {"voice_name": voice_name}))
        if upload_targets:
            tasks.append(TaskSpec(TaskType.UPLOAD))

        payload = {
            "audio_input": audio_input,
            "voice_name": voice_name,
            "transcript": transcript,
            "max_duration_s": max_duration_s,
            "upload_targets": upload_targets
        }
        return cls(
            parcel_id=p_id,
            tasks=tasks,
            payload=payload,
            return_artifacts=return_artifacts or (["voice_breeze"] if not upload_targets else None),
            upload_targets=upload_targets
        )

    @classmethod
    def standalone_asr(
        cls,
        audio_input: Union[str, bytes, np.ndarray, List[float]],
        output_format: str = "all",
        upload_targets: Optional[Dict[str, str]] = None,
        return_artifacts: Optional[List[str]] = None,
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """Constructs a standalone ASR transcription parcel."""
        p_id = parcel_id or f"parcel_asr_{int(time.time() * 1000)}"
        tasks = [TaskSpec(TaskType.ASR, {"output_format": output_format})]
        if upload_targets:
            tasks.append(TaskSpec(TaskType.UPLOAD))
        return cls(
            parcel_id=p_id,
            tasks=tasks,
            payload={"audio_input": audio_input, "upload_targets": upload_targets},
            return_artifacts=return_artifacts,
            upload_targets=upload_targets
        )

    @classmethod
    def standalone_enhance(
        cls,
        audio_input: Union[str, bytes, np.ndarray, List[float]],
        chunk_seconds: float = 10.0,
        master: bool = True,
        upload_targets: Optional[Dict[str, str]] = None,
        return_artifacts: Optional[List[str]] = None,
        parcel_id: Optional[str] = None
    ) -> 'Parcel':
        """Constructs a standalone Enhancer parcel."""
        p_id = parcel_id or f"parcel_enh_{int(time.time() * 1000)}"
        tasks = [TaskSpec(TaskType.ENHANCE, {"chunk_seconds": chunk_seconds})]
        if master:
            tasks.append(TaskSpec(TaskType.MASTER, {"preset": "optionA_rich_bass"}))
        if upload_targets:
            tasks.append(TaskSpec(TaskType.UPLOAD))
        return cls(
            parcel_id=p_id,
            tasks=tasks,
            payload={"audio_input": audio_input, "upload_targets": upload_targets},
            return_artifacts=return_artifacts,
            upload_targets=upload_targets
        )


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
    artifact_bytes: Dict[str, bytes] = field(default_factory=dict)       # Requested or fallback deliverable files in memory
    upload_results: Dict[str, Dict[str, Any]] = field(default_factory=dict) # Direct cloud bucket upload status
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

    def validate_parcel(self, parcel: Parcel) -> None:
        """Validates that all requested tasks can be fulfilled by the running engine instance."""
        self.capabilities.validate_parcel(parcel)

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

            # If payload provides an initial audio input, pre-load it into memory with strict 10s ceiling for voice registration
            if "audio_input" in parcel.payload and parcel.payload["audio_input"] is not None:
                inp = parcel.payload["audio_input"]
                task_types = [t.task_type for t in parcel.tasks]
                max_dur = parcel.payload.get("max_duration_s")
                if max_dur is None and TaskType.VOICE_REGISTER in task_types:
                    max_dur = 10.0

                report("audio_ingest", 0.02, f"Ingesting audio input (Max ceiling: {max_dur or 'Unlimited'}s)...")
                current_pcm, current_sr = load_any_audio(inp, target_sr=None, max_duration_s=max_dur)
                report("audio_ingest", 0.04, f"Ingested {len(current_pcm)/float(current_sr):.2f}s audio @ {current_sr} Hz.")

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

                # ------------------- 8. DIRECT CLOUD BUCKET UPLOAD -------------------
                elif t_type == TaskType.UPLOAD:
                    upload_targets = parcel.payload.get("upload_targets") or parcel.upload_targets or {}
                    if not upload_targets:
                        report("upload", 0.95, "No upload targets specified in parcel; skipping upload.")
                        continue

                    report("upload", 0.95, f"Uploading {len(upload_targets)} artifact(s) directly to cloud storage...")
                    upload_results: Dict[str, Dict[str, Any]] = {}
                    any_upload_failed = False

                    def get_artifact(name: str) -> Tuple[Optional[bytes], str]:
                        if name == "audio_mp3" and current_pcm is not None:
                            return encode_pcm_to_mp3_bytes(current_pcm, current_sr), "audio/mpeg"
                        elif name == "audio_wav" and current_pcm is not None:
                            return encode_pcm_to_wav_bytes(current_pcm, current_sr), "audio/wav"
                        elif name == "subtitles_srt" and result.subtitles_srt:
                            return result.subtitles_srt.encode("utf-8"), "application/x-subrip"
                        elif name == "transcript_txt" and result.transcript:
                            return result.transcript.encode("utf-8"), "text/plain; charset=utf-8"
                        elif name == "voice_breeze" and result.voice_bytes:
                            return result.voice_bytes, "application/octet-stream"
                        elif name == "raw_audio_wav" and result.raw_audio_pcm is not None:
                            return encode_pcm_to_wav_bytes(result.raw_audio_pcm, 24000), "audio/wav"
                        return None, "application/octet-stream"

                    for art_name, target_url in upload_targets.items():
                        art_bytes, mime = get_artifact(art_name)
                        if art_bytes is None:
                            upload_results[art_name] = {
                                "success": False,
                                "error": f"Artifact '{art_name}' was not produced by parcel tasks."
                            }
                            any_upload_failed = True
                            continue

                        report("upload", 0.97, f"PUT {art_name} ({len(art_bytes)/1024:.1f} KB) -> {target_url[:50]}...")
                        try:
                            req = urllib.request.Request(
                                target_url,
                                data=art_bytes,
                                headers={
                                    "Content-Type": mime,
                                    "Content-Length": str(len(art_bytes))
                                },
                                method="PUT"
                            )
                            with urllib.request.urlopen(req, timeout=30.0) as resp:
                                status = resp.status
                                if status in (200, 201, 204):
                                    upload_results[art_name] = {
                                        "success": True,
                                        "status": status,
                                        "bytes": len(art_bytes)
                                    }
                                else:
                                    upload_results[art_name] = {
                                        "success": False,
                                        "status": status,
                                        "error": f"HTTP {status}"
                                    }
                                    any_upload_failed = True
                        except Exception as up_err:
                            upload_results[art_name] = {
                                "success": False,
                                "error": str(up_err)
                            }
                            any_upload_failed = True

                    result.upload_results = upload_results
                    metadata["upload_results"] = upload_results
                    stage_timings["upload"] = time.time() - task_t0

                    # Transactional Fallback: If ANY upload failed, attach all artifact bytes to result
                    if any_upload_failed:
                        report("upload", 0.99, "⚠️ One or more uploads failed. Triggering transactional fallback (attaching raw artifact bytes)...")
                        for art_name in upload_targets.keys():
                            art_bytes, _ = get_artifact(art_name)
                            if art_bytes:
                                result.artifact_bytes[art_name] = art_bytes

            # -------------------------------------------------------------
            # FINAL RESULT ASSEMBLY (DYNAMIC ARTIFACTS & IN-MEMORY BUFFERS)
            # -------------------------------------------------------------
            result.audio_pcm = current_pcm
            result.sample_rate = current_sr
            result.duration_s = round(len(current_pcm) / float(current_sr), 2) if current_pcm is not None else 0.0
            result.stage_timings = stage_timings
            result.metadata = metadata

            # Populate requested in-memory artifacts if return_artifacts was specified
            requested_artifacts = parcel.return_artifacts or []
            if requested_artifacts:
                for art_name in requested_artifacts:
                    if art_name not in result.artifact_bytes:
                        if art_name == "audio_mp3" and current_pcm is not None:
                            result.artifact_bytes["audio_mp3"] = encode_pcm_to_mp3_bytes(current_pcm, current_sr)
                        elif art_name == "audio_wav" and current_pcm is not None:
                            result.artifact_bytes["audio_wav"] = encode_pcm_to_wav_bytes(current_pcm, current_sr)
                        elif art_name == "subtitles_srt" and result.subtitles_srt:
                            result.artifact_bytes["subtitles_srt"] = result.subtitles_srt.encode("utf-8")
                        elif art_name == "transcript_txt" and result.transcript:
                            result.artifact_bytes["transcript_txt"] = result.transcript.encode("utf-8")
                        elif art_name == "voice_breeze" and result.voice_bytes:
                            result.artifact_bytes["voice_breeze"] = result.voice_bytes
                        elif art_name == "raw_audio_wav" and result.raw_audio_pcm is not None:
                            result.artifact_bytes["raw_audio_wav"] = encode_pcm_to_wav_bytes(result.raw_audio_pcm, 24000)

            report("completed", 1.0, f"Parcel {parcel.parcel_id} processed successfully in {time.time()-t_start_total:.2f}s!")
            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            report("failed", 1.0, f"Parcel processing failed: {str(e)}")
            return result
