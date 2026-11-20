#!/usr/bin/env python3
"""
Breeze-TTS Studio: Decoupled Gradio Web Application Client
Provides in-browser Text-to-Speech, Zero-Shot Voice Cloning, Human-in-the-Loop Voice Registration (.breeze),
Speech-to-Speech Voice Conversion (Mode 4), and Subtitle/Transcription tools.
Strictly decoupled from the engine: consumes public bz_ggml APIs and respects all engine contracts.
"""

import argparse
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from typing import Optional, Tuple, List, Dict, Any, Generator
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bz_ggml import (
    DualInstanceCluster, UserTask, VoiceConversionTask,
    load_breeze_voice, save_breeze_voice,
    serialize_breeze_voice, deserialize_breeze_voice,
    EngineOrchestrator, EngineCapabilities,
    TaskType, TaskSpec, Parcel, ParcelResult,
    save_pcm_to_mp3, save_pcm_to_wav
)
from bz_ggml.asr import ASRService
from bz_ggml.audio_pipeline import (
    split_text_into_smart_chunks, stitch_audio_chunks,
    apply_resemble_enhancer, apply_studio_normalization,
    convert_to_mp3_256k, generate_1word_srt_file,
    format_chunk_preview_markdown
)

import gradio as gr

# Global Configuration Dictionary (Populated by CLI args)
APP_CONFIG: Dict[str, Any] = {
    "model_path": "breeze-tts-2-q8_0.gguf",
    "lib_path": None,
    "q4_model_path": None,
    "enable_q4_burst": False,
    "q4_threshold": 10,
    "vocoder_mode": "on",
    "precision": "int8",
    "enable_enhancer": True,
    "enable_asr": True,
    "enhancer_chunk_sec": 10.0,
    "asr_model": "turbo",
    "asr_device": 0,
    "enhancer_device": 1,
    "voices_dir": "data/voices",
    "outputs_dir": "/kaggle/working/data/outputs" if os.path.exists("/kaggle/working") else "data/outputs",
}

cluster_instance: Optional[DualInstanceCluster] = None
asr_service_instance: Optional[ASRService] = None
orchestrator_instance: Optional[EngineOrchestrator] = None


def list_deliverable_files() -> List[str]:
    """Returns sorted list of saved deliverable files (.mp3, .srt, .wav) from client outputs directory."""
    out_dir = APP_CONFIG.get("outputs_dir", "data/outputs")
    if not os.path.exists(out_dir):
        return []
    files = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".mp3") or f.endswith(".srt") or f.endswith(".wav")
    ]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def get_cluster() -> DualInstanceCluster:
    """Initializes or returns the singleton DualInstanceCluster using APP_CONFIG."""
    global cluster_instance
    if cluster_instance is None:
        model_path = APP_CONFIG["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"[Gradio Studio] Initializing DualInstanceCluster with Model: {model_path}")
        print(f"  • Precision: {APP_CONFIG.get('precision', 'int8')} | Vocoder Mode: {APP_CONFIG['vocoder_mode']} | Q4 Burst: {APP_CONFIG['enable_q4_burst']}")
        cluster_instance = DualInstanceCluster(
            model_path=model_path,
            lib_path=APP_CONFIG["lib_path"],
            q4_model_path=APP_CONFIG["q4_model_path"],
            enable_q4_burst=APP_CONFIG["enable_q4_burst"],
            q4_threshold=APP_CONFIG["q4_threshold"],
            enable_vocoder=True,
            vocoder_mode=APP_CONFIG["vocoder_mode"],
            precision=APP_CONFIG.get("precision", "int8"),
            enable_enhancer=APP_CONFIG.get("enable_enhancer", True),
            enable_asr=APP_CONFIG.get("enable_asr", True)
        )
    return cluster_instance


def get_asr_service() -> Optional[ASRService]:
    """Initializes or returns the singleton ASRService pinned to configured GPU."""
    global asr_service_instance
    if not APP_CONFIG.get("enable_asr", True):
        return None
    if asr_service_instance is None:
        asr_service_instance = ASRService(
            model_size=APP_CONFIG["asr_model"],
            cuda_device=APP_CONFIG["asr_device"],
            compute_type="float16"
        )
    return asr_service_instance


def get_orchestrator() -> EngineOrchestrator:
    """Initializes or returns the singleton EngineOrchestrator."""
    global orchestrator_instance
    if orchestrator_instance is None:
        cluster = get_cluster()
        asr = get_asr_service()
        caps = EngineCapabilities(
            precision=APP_CONFIG.get("precision", "int8"),
            enable_q4_surge=APP_CONFIG.get("enable_q4_burst", False),
            vocoder_mode=APP_CONFIG.get("vocoder_mode", "on"),
            enable_enhancer=APP_CONFIG.get("enable_enhancer", True),
            enable_asr=APP_CONFIG.get("enable_asr", True),
            chunk_seconds_enhancer=APP_CONFIG.get("enhancer_chunk_sec", 10.0),
            asr_model_size=APP_CONFIG.get("asr_model", "turbo"),
            asr_device_index=APP_CONFIG.get("asr_device", 0),
            enhancer_device_index=APP_CONFIG.get("enhancer_device", 1)
        )
        orchestrator_instance = EngineOrchestrator(
            cluster=cluster,
            asr_service=asr,
            capabilities=caps
        )
    return orchestrator_instance


def load_audio_24k(audio_path: str) -> List[float]:
    """Loads an audio file and converts to 24 kHz mono float32 in [-1.0, 1.0]."""
    try:
        import torchaudio
        sig, sr = torchaudio.load(audio_path)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        if sr != 24000:
            sig = torchaudio.functional.resample(sig, sr, 24000)
        return sig.squeeze().cpu().numpy().astype(np.float32).tolist()
    except Exception:
        pass

    try:
        import soundfile as sf
        import scipy.signal
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 24000:
            num_samples = int(len(data) * 24000 / sr)
            data = scipy.signal.resample(data, num_samples)
        return data.astype(np.float32).tolist()
    except Exception:
        pass

    import wave
    with wave.open(audio_path, 'rb') as wf:
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        if width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)
        if sr != 24000:
            import scipy.signal
            num_samples = int(len(data) * 24000 / sr)
            data = scipy.signal.resample(data, num_samples)
        return data.astype(np.float32).tolist()


def list_saved_voices() -> List[str]:
    """Scans APP_CONFIG['voices_dir'] and returns list of available voice personas."""
    v_dir = APP_CONFIG["voices_dir"]
    os.makedirs(v_dir, exist_ok=True)
    voices = ["(None / Default Voice)"]
    for f in sorted(os.listdir(v_dir)):
        if f.endswith(".breeze"):
            name = f[:-7].replace("_", " ").title()
            voices.append(name)
    return voices


def get_voice_file_path(voice_display_name: str) -> Optional[str]:
    """Resolves a voice display name to its absolute .breeze file path."""
    if not voice_display_name or voice_display_name.startswith("(None"):
        return None
    raw_name = voice_display_name.lower().replace(" ", "_")
    path = os.path.join(APP_CONFIG["voices_dir"], f"{raw_name}.breeze")
    return path if os.path.exists(path) else None


def transcribe_audio_sample(audio_path: Optional[str]) -> str:
    """Transcribes an audio sample on GPU 1 using faster-whisper with VAD filtering."""
    if not audio_path or not os.path.exists(audio_path):
        return "Error: Please upload or record an audio file first."
    try:
        asr = get_asr_service()
        transcript = asr.transcribe(audio_path, vad_filter=True)
        return transcript.strip()
    except Exception as e:
        return f"ASR Transcription Error: {str(e)}"


def transcribe_subtitles_tool(audio_path: Optional[str]) -> Tuple[str, str]:
    """Transcribes audio of arbitrary length into full text and SRT timestamped subtitles."""
    if not audio_path or not os.path.exists(audio_path):
        return "Error: Please upload an audio file.", ""
    try:
        asr = get_asr_service()
        transcript, srt_content, _ = asr.transcribe(audio_path, vad_filter=True, return_subtitles=True)
        return transcript, srt_content
    except Exception as e:
        return f"ASR Error: {str(e)}", ""


def register_voice_file(
    voice_name: str,
    audio_path: Optional[str],
    enhance: bool = True
) -> Tuple[Optional[str], str, Any, Any, Any]:
    """
    Submits a Voice Persona Registration parcel [TRIM <= 10s, ENHANCE (if checked), ASR, VOICE_REGISTER]
    to the Engine Orchestrator, receives in-memory .breeze bytes, saves to voices_dir, and updates UI dropdowns.
    Zero manual transcript required (Whisper Turbo v3 automatically transcribes).
    """
    if not voice_name.strip():
        return None, "❌ Error: Voice name is required.", gr.update(), gr.update(), gr.update()
    if not audio_path or not os.path.exists(audio_path):
        return None, "❌ Error: Audio sample file is required.", gr.update(), gr.update(), gr.update()

    try:
        orchestrator = get_orchestrator()
        parcel = Parcel.voice_registration(
            audio_input=audio_path,
            voice_name=voice_name.strip(),
            enhance=bool(enhance),
            max_duration_s=10.0
        )
        t0 = time.time()
        res = orchestrator.dispatch(parcel)
        if not res.success or res.voice_bytes is None:
            raise RuntimeError(res.error_message or "Failed to encode voice container")

        container_bytes = res.voice_bytes
        enc_time = time.time() - t0

        # Client-side storage: save bytes to configured voices_dir
        v_dir = APP_CONFIG["voices_dir"]
        os.makedirs(v_dir, exist_ok=True)
        safe_name = "".join(c for c in voice_name.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
        breeze_filename = f"{safe_name.lower().replace(' ', '_')}.breeze"
        breeze_path = os.path.join(v_dir, breeze_filename)

        with open(breeze_path, "wb") as f:
            f.write(container_bytes)

        # Also create tempfile copy for Gradio file widget serving
        temp_export = os.path.join(tempfile.gettempdir(), breeze_filename)
        with open(temp_export, "wb") as f:
            f.write(container_bytes)

        # Read back container metadata for display
        meta = deserialize_breeze_voice(container_bytes)
        file_kb = len(container_bytes) / 1024.0

        msg = (
            f"✅ **Voice Registered Successfully!**\n\n"
            f"• **Voice Persona**: `{voice_name.strip()}`\n"
            f"• **Auto-Transcribed Text**: \"{res.transcript}\"\n"
            f"• **Enhanced on GPU 1**: {'Yes (Resemble 44.1 kHz)' if enhance else 'No (Direct 24 kHz)'}\n"
            f"• **Acoustic Frames**: {meta['frames']} ({meta['frames'] * 0.08:.2f}s reference audio - strictly capped <= 10s)\n"
            f"• **Discrete Tokens**: {len(meta['codes'])} tokens (16 codebooks @ 12.5 Hz)\n"
            f"• **Encode Time**: {enc_time:.2f}s (Pinned to GPU 0 Reference Encoder Pool)\n"
            f"• **Saved In Registry**: `{breeze_path}` ({file_kb:.1f} KB)\n"
            f"• **Instant Availability**: Now selectable in all Cloning & TTS dropdowns with 0 ms load!"
        )

        updated_choices = list_saved_voices()
        return (
            temp_export,
            msg,
            gr.update(choices=updated_choices, value=voice_name.strip().title()),
            gr.update(choices=updated_choices, value=voice_name.strip().title()),
            gr.update(choices=updated_choices, value=voice_name.strip().title())
        )
    except Exception as e:
        return None, f"❌ Voice Registration Failed: {str(e)}", gr.update(), gr.update(), gr.update()


def tts_generate(
    text: str,
    instruction: str,
    selected_voice: str,
    delivery_mode: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    max_steps: int
) -> Generator[Tuple[Any, Any, Any, str], None, None]:
    if not text.strip():
        yield gr.skip(), None, gr.update(visible=False), "❌ Error: Text input is empty."
        return

    ref_codes = None
    ref_frames = 0
    ref_text = None
    voice_path = get_voice_file_path(selected_voice)
    if voice_path:
        v_data = load_breeze_voice(voice_path)
        ref_codes = v_data["codes"]
        ref_frames = v_data["frames"]
        ref_text = v_data["text"]

    voice_label = selected_voice if voice_path else "Default Native Speaker"
    orchestrator = get_orchestrator()

    parcel = Parcel.tts_pipeline(
        script_text=text.strip(),
        voice_name=selected_voice,
        instruction=instruction,
        cfg_scale=float(cfg_scale),
        temperature=float(temperature),
        seed=int(seed),
        enhance=False,
        master=True,
        srt=False,
        ref_codes=ref_codes,
        ref_frames=ref_frames,
        ref_text=ref_text,
        parcel_id=f"tts_single_{int(time.time()*1000)}"
    )

    yield gr.skip(), None, gr.update(visible=False), f"⏳ Synthesizing audio on Dual GPU Islands ({voice_label})..."

    t0 = time.time()
    res = orchestrator.dispatch(parcel)
    wall = time.time() - t0

    if not res.success or res.audio_pcm is None:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Synthesis failed: {res.error_message}"
        return

    out_wav = os.path.join(tempfile.gettempdir(), f"{parcel.parcel_id}.wav")
    save_pcm_to_wav(res.audio_pcm, res.sample_rate, out_wav)

    stats_msg = (
        f"✅ **Synthesis Complete** | Studio Master\n\n"
        f"• **Voice Persona**: `{voice_label}`\n"
        f"• **Audio Length**: {res.duration_s:.2f}s ({len(res.audio_pcm):,} samples @ {res.sample_rate} Hz)\n"
        f"• **Compute Time**: {wall:.2f}s\n"
        f"• **Master File**: `{os.path.basename(out_wav)}` ({os.path.getsize(out_wav)/1024:.1f} KB)"
    )
    yield gr.skip(), out_wav, gr.update(value=out_wav, visible=True), stats_msg


def voice_clone_generate(
    text: str,
    selected_voice: str,
    instruction: str,
    delivery_mode: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    max_steps: int
) -> Generator[Tuple[Any, Any, Any, str], None, None]:
    if not text.strip():
        yield gr.skip(), None, gr.update(visible=False), "❌ Error: Target text is empty."
        return

    voice_path = get_voice_file_path(selected_voice)
    if not voice_path:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Error: Voice persona '{selected_voice}' not found in registry."
        return

    try:
        v_data = load_breeze_voice(voice_path)
        ref_codes = v_data["codes"]
        ref_frames = v_data["frames"]
        ref_text = v_data["text"]
    except Exception as e:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Error loading voice container: {str(e)}"
        return

    orchestrator = get_orchestrator()
    parcel = Parcel.tts_pipeline(
        script_text=text.strip(),
        voice_name=selected_voice,
        instruction=instruction,
        cfg_scale=float(cfg_scale),
        temperature=float(temperature),
        seed=int(seed),
        enhance=False,
        master=True,
        srt=False,
        ref_codes=ref_codes,
        ref_frames=ref_frames,
        ref_text=ref_text,
        parcel_id=f"clone_{int(time.time()*1000)}"
    )

    yield gr.skip(), None, gr.update(visible=False), f"⏳ Synthesizing cloned voice on Dual GPU Islands (`{selected_voice}`)..."

    t0 = time.time()
    res = orchestrator.dispatch(parcel)
    wall = time.time() - t0

    if not res.success or res.audio_pcm is None:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Synthesis failed: {res.error_message}"
        return

    out_wav = os.path.join(tempfile.gettempdir(), f"{parcel.parcel_id}.wav")
    save_pcm_to_wav(res.audio_pcm, res.sample_rate, out_wav)

    stats_msg = (
        f"✅ **Cloning Complete** | Studio Master\n\n"
        f"• **Cloned Persona**: `{selected_voice}`\n"
        f"• **Audio Length**: {res.duration_s:.2f}s ({len(res.audio_pcm):,} samples @ {res.sample_rate} Hz)\n"
        f"• **Compute Time**: {wall:.2f}s\n"
        f"• **Master File**: `{os.path.basename(out_wav)}` ({os.path.getsize(out_wav)/1024:.1f} KB)"
    )
    yield gr.skip(), out_wav, gr.update(value=out_wav, visible=True), stats_msg


def voice_convert_generate(
    source_audio: Optional[str],
    selected_voice: str,
    ref_audio: Optional[str],
    ref_text: str,
    keep_acoustic: int,
    cfg_scale: float,
    seed: int
) -> Tuple[Optional[str], str]:
    if not source_audio:
        return None, "Error: Source audio file is required."

    voice_path = get_voice_file_path(selected_voice)
    ref_codes = None
    ref_frames = 0
    final_ref_text = None

    if voice_path:
        v_data = load_breeze_voice(voice_path)
        ref_codes = v_data["codes"]
        ref_frames = v_data["frames"]
        final_ref_text = v_data["text"]
    elif ref_audio and os.path.exists(ref_audio):
        cluster = get_cluster()
        ref_pcm = load_audio_24k(ref_audio)
        ref_codes, ref_frames = cluster.encode_audio(ref_pcm, gpu_id=0)
        final_ref_text = ref_text.strip() if ref_text else None
    else:
        return None, "Error: Please select a registered voice or provide target reference audio."

    try:
        cluster = get_cluster()
        src_pcm = load_audio_24k(source_audio)
        src_codes, src_frames = cluster.encode_audio(src_pcm, gpu_id=0)
        if src_frames <= 0 or not src_codes:
            return None, "❌ Error: Could not encode source audio frames."

        task = VoiceConversionTask(
            id=int(time.time() * 1000) % 100000,
            src_codes=src_codes,
            src_frames=src_frames,
            ref_codes=ref_codes,
            ref_frames=ref_frames,
            ref_text=final_ref_text,
            cfg_scale=float(cfg_scale),
            keep_acoustic=int(keep_acoustic),
            seed=int(seed),
            client_has_vocoder=False,
            stream_pcm=False
        )

        audio_samples, tokens, stats = cluster.convert_voice_task(task)
        if not audio_samples:
            return None, "Error: Voice conversion produced no samples."

        out_wav = os.path.join(tempfile.gettempdir(), f"converted_voice_{task.id}.wav")
        import wave
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            pcm_np = np.array(audio_samples, dtype=np.float32)
            pcm_int16 = (np.clip(pcm_np, -1.0, 1.0) * 32767.0).astype(np.int16)
            w.writeframes(pcm_int16.tobytes())

        target_desc = selected_voice if voice_path else (os.path.basename(ref_audio) if ref_audio else "Reference Audio")
        stats_msg = (
            f"✅ **Voice Conversion Complete (Mode 4)**\n\n"
            f"• **Source Audio**: `{os.path.basename(source_audio)}` ({src_frames * 0.08:.2f}s, {src_frames} frames)\n"
            f"• **Target Persona**: `{target_desc}`\n"
            f"• **Wall Time**: {stats['compute_wall_s']:.2f}s (RTF: {stats['rtf']:.3f}x)\n"
            f"• **Acoustic Codebooks Kept**: {keep_acoustic}/15\n"
            f"• **Worker Island**: {stats['worker']}"
        )
        return out_wav, stats_msg
    except Exception as e:
        return None, f"❌ Error during voice conversion: {str(e)}"


def run_longform_pipeline_ui(
    script_text: str,
    selected_voice: str,
    instruction: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    pause_ms: int,
    target_words: int,
    max_words: int,
    apply_enhance: bool,
    apply_master: bool,
    apply_srt: bool
) -> Generator[Tuple[str, Optional[str], Optional[str], Optional[str], Any, Any, Any, Any], None, None]:
    if not script_text.strip():
        yield "❌ Error: Script text is empty.", None, None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), list_deliverable_files()
        return

    out_dir = APP_CONFIG["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)

    ref_codes = None
    ref_frames = 0
    ref_text = None
    voice_path = get_voice_file_path(selected_voice)
    if voice_path:
        v_data = load_breeze_voice(voice_path)
        ref_codes = v_data["codes"]
        ref_frames = v_data["frames"]
        ref_text = v_data["text"]

    job_id = f"prod_{int(time.time() * 1000)}"
    parcel = Parcel.tts_pipeline(
        script_text=script_text.strip(),
        voice_name=selected_voice,
        instruction=instruction,
        cfg_scale=float(cfg_scale),
        temperature=float(temperature),
        seed=int(seed),
        pause_ms=int(pause_ms),
        target_words=int(target_words),
        max_words=int(max_words),
        enhance=bool(apply_enhance),
        master=bool(apply_master),
        srt=bool(apply_srt),
        ref_codes=ref_codes,
        ref_frames=ref_frames,
        ref_text=ref_text,
        parcel_id=job_id
    )

    update_queue = queue.Queue()

    def progress_callback(stage: str, progress: float, msg: str):
        update_queue.put({"stage": stage, "progress": progress, "msg": msg})

    orchestrator = get_orchestrator()

    def worker_thread():
        try:
            res = orchestrator.dispatch(parcel, progress_callback=progress_callback)
            update_queue.put({"stage": "completed", "result": res})
        except Exception as e:
            update_queue.put({"stage": "failed", "error": str(e)})

    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()

    last_log = "🚀 Dispatching Parcel to Engine Orchestrator..."
    raw_mp3_path = None
    mastered_mp3_path = None
    srt_content = ""
    srt_file_path = None

    while True:
        try:
            item = update_queue.get(timeout=0.25)
            stage = item.get("stage")

            if stage == "completed":
                res: ParcelResult = item["result"]
                if not res.success:
                    yield (
                        f"❌ **Engine Error**: {res.error_message}",
                        None, None, "",
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        list_deliverable_files()
                    )
                    break

                # Client-Side Persistence: Save in-memory results directly to outputs directory
                if res.raw_audio_pcm is not None and len(res.raw_audio_pcm) > 0:
                    raw_mp3_path = os.path.join(out_dir, f"{job_id}_raw.mp3")
                    save_pcm_to_mp3(res.raw_audio_pcm, 24000, raw_mp3_path)

                if res.audio_pcm is not None and len(res.audio_pcm) > 0:
                    mastered_mp3_path = os.path.join(out_dir, f"{job_id}_mastered.mp3")
                    save_pcm_to_mp3(res.audio_pcm, res.sample_rate, mastered_mp3_path)

                if res.subtitles_srt:
                    srt_content = res.subtitles_srt
                    srt_file_path = os.path.join(out_dir, f"{job_id}_transcript_1word.srt")
                    with open(srt_file_path, "w", encoding="utf-8") as f:
                        f.write(srt_content)

                mins = int(res.duration_s // 60)
                secs = int(res.duration_s % 60)
                summary = (
                    f"🎉 **Parcel Pipeline Completed Successfully!**\n\n"
                    f"• **Parcel ID**: `{res.parcel_id}`\n"
                    f"• **Cohesive Chunks**: {res.chunks_count} chunks synthesized symmetrically across Dual GPUs\n"
                    f"• **Total Audio**: {res.duration_s:.2f}s ({mins}m {secs}s @ {res.sample_rate} Hz)\n"
                    f"• **Raw Voiceover**: `{os.path.basename(raw_mp3_path) if raw_mp3_path else 'N/A'}` (256k MP3)\n"
                    f"• **Mastered Voiceover**: `{os.path.basename(mastered_mp3_path) if mastered_mp3_path else 'N/A'}` (Broadcast Standard)\n"
                    f"• **Word Subtitles**: `{os.path.basename(srt_file_path) if srt_file_path else 'Disabled'}`\n"
                    f"• **Stage Timings**: {', '.join(f'{k}: {v:.2f}s' for k, v in res.stage_timings.items())}"
                )

                saved_files = list_deliverable_files()
                yield (
                    summary,
                    raw_mp3_path,
                    mastered_mp3_path,
                    srt_content,
                    gr.update(value=mastered_mp3_path, visible=bool(mastered_mp3_path)),
                    gr.update(value=raw_mp3_path, visible=bool(raw_mp3_path)),
                    gr.update(value=srt_file_path, visible=bool(srt_file_path)),
                    saved_files
                )
                break

            elif stage == "failed":
                err = item.get("error", "Unknown error")
                yield (
                    f"❌ **Pipeline Failure**: {err}",
                    raw_mp3_path,
                    mastered_mp3_path,
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    list_deliverable_files()
                )
                break

            else:
                last_log = item.get("msg", last_log)
                pct = item.get("progress", 0.0)
                yield (
                    f"⏳ **[{stage.upper()}] ({int(pct*100)}%)**: {last_log}",
                    raw_mp3_path,
                    mastered_mp3_path,
                    srt_content,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.skip()
                )

        except queue.Empty:
            if not thread.is_alive():
                break


def build_app() -> gr.Blocks:
    custom_css = """
    .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .status-badge { font-weight: bold; color: #10b981; }
    """

    initial_voices = list_saved_voices()

    with gr.Blocks(title="Breeze-TTS Studio", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ Breeze-TTS Studio & Voice Lab
            ### High-Fidelity Neural Speech Synthesis, Zero-Shot Voice Cloning & Subtitles
            *32-Frame Bit-Exact Stateful Vocoder | Whisper ASR Microservice | Portable `.breeze` Containers*
            """
        )

        with gr.Tabs():
            # ---------------- TAB 0: LONG-FORM PRODUCTION PIPELINE ----------------
            with gr.TabItem("🎬 Long-Form Studio (Auto-Chunk, Multi-GPU Parallel & Mastering)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
                            ### 🎬 Long-Form Script Engine & Studio Mastering
                            *Unlimited Length Scripts • Semantic Chunking • Dual-GPU Continuous Batching • Resemble Enhancer • Studio Broadcast Normalization • 1-Word SRT*
                            """
                        )
                        lf_script = gr.Textbox(
                            label="Full Script to Synthesize (Unlimited Length)",
                            placeholder="Paste your long-form script here (articles, documentary scripts, stories)... It will be chunked intelligently at natural paragraph and sentence boundaries.",
                            lines=10,
                            value=(
                                "Welcome to the full production pipeline demonstration.\n\n"
                                "Our system features an autonomous dual-island architecture with zero PCIe bottlenecks. "
                                "Each Tesla T4 GPU operates its own resident generator and dedicated streaming vocoder in VRAM.\n\n"
                                "When long scripts are submitted, the intelligent semantic chunker partitions the text into cohesive "
                                "speech units without ever slicing sentences or paragraphs in half. All chunks are dispatched concurrently "
                                "across both GPUs using multi-session continuous batching, saturating hardware capacity for maximum throughput."
                            )
                        )
                        with gr.Row():
                            lf_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0],
                                label="Voice Persona (From Voices Registry)",
                                scale=4
                            )
                            lf_refresh_voice = gr.Button("🔄", scale=1)

                        lf_refresh_voice.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[lf_voice_dropdown]
                        )

                        lf_instruction = gr.Textbox(
                            label="Speaker Style / Narration Direction",
                            value="Speak with natural clarity, warm authoritative pacing, and professional broadcast presence."
                        )

                        with gr.Row():
                            lf_cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Guidance Scale (1.0 = Clean, 4.0 = Expressive)")
                            lf_temp = gr.Slider(0.05, 1.2, value=0.25, step=0.05, label="Temperature (Sampling Randomness)")
                            lf_seed = gr.Number(value=42, label="Base Random Seed", precision=0)
                            lf_pause = gr.Slider(100, 600, value=200, step=25, label="Inter-Chunk Breath Pause (ms)")

                        with gr.Accordion("✂️ Intelligent Chunking Controls & Visual Preview", open=True):
                            with gr.Row():
                                lf_target_words = gr.Slider(30, 200, value=80, step=5, label="Target Words per Chunk (Ideal: 70-90w)")
                                lf_max_words = gr.Slider(50, 250, value=120, step=5, label="Max Words per Chunk (Ceiling: 120-150w)")
                            lf_preview_btn = gr.Button("🔍 Preview Chunks & Word Breakdown")
                            lf_preview_box = gr.Markdown("Paste or type your script above and click **Preview Chunks** to inspect how the parser partitions your text.")

                        lf_preview_btn.click(
                            fn=lambda text, tw, mw: format_chunk_preview_markdown(text, target_words=int(tw), max_words=int(mw)),
                            inputs=[lf_script, lf_target_words, lf_max_words],
                            outputs=[lf_preview_box]
                        )

                        gr.Markdown("#### 🎛️ Post-Processing & Mastering Controls")
                        with gr.Row():
                            lf_chk_enhance = gr.Checkbox(value=True, label="✨ Apply Resemble Enhancer (44.1 kHz Denoise & Upsample)")
                            lf_chk_master = gr.Checkbox(value=True, label="🎚️ Apply Gentle Studio Mastering (swift_history Winning Preset)")
                            lf_chk_srt = gr.Checkbox(value=True, label="📝 Generate 1-Word per Line SRT Subtitles")

                        lf_submit_btn = gr.Button("🚀 Dispatch Long-Form Production Pipeline", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        lf_status_box = gr.Markdown("Ready to process script.")
                        lf_audio_raw = gr.Audio(label="1️⃣ Raw Stitched Voiceover (256k MP3)", type="filepath")
                        lf_audio_mastered = gr.Audio(label="2️⃣ Enhanced & Mastered Voiceover (256k MP3)", type="filepath")
                        lf_srt_box = gr.Textbox(label="3️⃣ 1-Word per Line SRT Subtitles Preview", lines=7)

                        with gr.Row():
                            lf_down_mastered = gr.DownloadButton("📥 Download Mastered MP3 (256k)", visible=False, variant="primary")
                            lf_down_raw = gr.DownloadButton("📥 Download Raw MP3", visible=False)
                            lf_down_srt = gr.DownloadButton("📥 Download 1-Word SRT", visible=False)

                        gr.Markdown("### 📂 Saved Deliverables History")
                        lf_history_refresh = gr.Button("🔄 Refresh Saved Files")
                        lf_history_files = gr.File(
                            label="Download Output Files (.mp3 / .srt)",
                            file_count="multiple",
                            value=list_deliverable_files()
                        )
                        lf_history_refresh.click(
                            fn=list_deliverable_files,
                            inputs=[],
                            outputs=[lf_history_files]
                        )

                lf_submit_btn.click(
                    fn=run_longform_pipeline_ui,
                    inputs=[
                        lf_script, lf_voice_dropdown, lf_instruction,
                        lf_cfg, lf_temp, lf_seed, lf_pause,
                        lf_target_words, lf_max_words,
                        lf_chk_enhance, lf_chk_master, lf_chk_srt
                    ],
                    outputs=[
                        lf_status_box, lf_audio_raw, lf_audio_mastered, lf_srt_box,
                        lf_down_mastered, lf_down_raw, lf_down_srt,
                        lf_history_files
                    ]
                )

            # ---------------- TAB 1: TEXT-TO-SPEECH & VOICE DESIGN ----------------
            with gr.TabItem("✨ Text-to-Speech & Voice Design"):
                with gr.Row():
                    with gr.Column(scale=3):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Type or paste the script you want synthesized...",
                            lines=4,
                            value="Welcome to Breeze-TTS Studio. Today we are demonstrating stateful streaming speech synthesis on dual GPUs."
                        )
                        tts_instruction = gr.Textbox(
                            label="Speaker Persona / Emotional Style Instruction",
                            placeholder="Describe tone, accent, pacing, or emotional style...",
                            value="Speak with a warm, authoritative, engaging documentary narration style."
                        )

                        with gr.Row():
                            tts_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0],
                                label="Voice Persona (From Voices Registry)",
                                scale=4
                            )
                            tts_refresh_btn = gr.Button("🔄", scale=1)

                        tts_refresh_btn.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[tts_voice_dropdown]
                        )

                        with gr.Row():
                            tts_delivery = gr.Radio(
                                choices=[
                                    "Real-Time Streaming (32-Frame Chunks)",
                                    "Studio Master (Full Single-Pass, Zero Clicks)"
                                ],
                                value="Real-Time Streaming (32-Frame Chunks)",
                                label="Vocoder Synthesis Mode"
                            )

                        with gr.Accordion("Advanced Generation Parameters", open=False):
                            with gr.Row():
                                tts_cfg = gr.Slider(1.0, 5.0, value=1.0, step=0.1, label="CFG Guidance Scale (1.0 = Pure TTS, 2.5 = (laugh)/(sigh), 4.0 = Voice Design)")
                                tts_temp = gr.Slider(0.1, 1.2, value=0.25, step=0.05, label="Temperature")
                                tts_seed = gr.Number(value=42, label="Random Seed")
                                tts_max = gr.Slider(100, 2040, value=1200, step=10, label="Max Output Frames (Up to 2,040 Hardware Limit)")

                        tts_btn = gr.Button("▶ Generate Audio", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        tts_stream_audio = gr.Audio(
                            label="🔊 Live Audio Stream (Real-Time Playback)",
                            streaming=True,
                            autoplay=True
                        )
                        tts_master_audio = gr.Audio(
                            label="🎵 Complete Master Audio Recording (Seekable Waveform & Replay)",
                            type="filepath",
                            show_download_button=True
                        )
                        tts_download_file = gr.File(
                            label="Download Master WAV (Single-File)",
                            file_types=[".wav"],
                            visible=False
                        )
                        tts_stats = gr.Markdown("Ready to synthesize.")

                tts_btn.click(
                    fn=tts_generate,
                    inputs=[
                        tts_text, tts_instruction, tts_voice_dropdown, tts_delivery,
                        tts_cfg, tts_temp, tts_seed, tts_max
                    ],
                    outputs=[tts_stream_audio, tts_master_audio, tts_download_file, tts_stats]
                )

            # ---------------- TAB 2: ZERO-SHOT VOICE CLONING ----------------
            with gr.TabItem("🎭 Zero-Shot Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=3):
                        clone_text = gr.Textbox(
                            label="Text to Speak in Cloned Voice",
                            placeholder="Enter the text you want the cloned speaker to say...",
                            lines=3,
                            value="This is a demonstration of zero-shot voice cloning using pre-registered voice containers."
                        )

                        with gr.Row():
                            clone_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0] if initial_voices else None,
                                label="Voice Persona (From Voices Registry)",
                                scale=4
                            )
                            clone_refresh_btn = gr.Button("🔄", scale=1)

                        clone_refresh_btn.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[clone_voice_dropdown]
                        )

                        clone_instruction = gr.Textbox(
                            label="Emotional Delivery Adjustment (Optional)",
                            placeholder="e.g., Speak with excitement and energy",
                            value="Speak clearly and naturally."
                        )

                        with gr.Row():
                            clone_delivery = gr.Radio(
                                choices=[
                                    "Real-Time Streaming (32-Frame Chunks)",
                                    "Studio Master (Full Single-Pass, Zero Clicks)"
                                ],
                                value="Real-Time Streaming (32-Frame Chunks)",
                                label="Vocoder Synthesis Mode"
                            )

                        with gr.Accordion("Advanced Voice Cloning Parameters", open=False):
                            with gr.Row():
                                clone_cfg = gr.Slider(1.0, 5.0, value=1.0, step=0.1, label="CFG Guidance Scale (1.0 = Exact Clone, 4.0 = Emotional Steering/Voice Edit)")
                                clone_temp = gr.Slider(0.1, 1.2, value=0.25, step=0.05, label="Temperature")
                                clone_seed = gr.Number(value=42, label="Random Seed")
                                clone_max = gr.Slider(100, 2040, value=1200, step=10, label="Max Output Frames (Up to 2,040 Hardware Limit)")

                        clone_btn = gr.Button("▶ Clone Voice & Synthesize", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        clone_stream_audio = gr.Audio(
                            label="🔊 Live Cloned Speech Stream (Real-Time Playback)",
                            streaming=True,
                            autoplay=True
                        )
                        clone_master_audio = gr.Audio(
                            label="🎵 Complete Master Audio Recording (Seekable Waveform & Replay)",
                            type="filepath",
                            show_download_button=True
                        )
                        clone_download_file = gr.File(
                            label="Download Master Cloned WAV (Single-File)",
                            file_types=[".wav"],
                            visible=False
                        )
                        clone_stats = gr.Markdown("Select a registered voice to begin.")

                clone_btn.click(
                    fn=voice_clone_generate,
                    inputs=[
                        clone_text, clone_voice_dropdown, clone_instruction, clone_delivery,
                        clone_cfg, clone_temp, clone_seed, clone_max
                    ],
                    outputs=[clone_stream_audio, clone_master_audio, clone_download_file, clone_stats]
                )

            # ---------------- TAB 3: VOICE REGISTRATION & MANAGEMENT ----------------
            with gr.TabItem("🎙️ Voice Registration & Management (.breeze)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
                            ### Human-in-the-Loop Voice Registration
                            1. **Record or Upload** reference audio (4–10 seconds).
                            2. **Click Auto-Transcribe** to run GPU 1 ASR (`faster-whisper-small`).
                            3. **Review & Correct** transcript in the text box below.
                            4. **Register**: The engine compiles 16-codebook tokens and returns in-memory `.breeze` bytes, saving to the registry.
                            """
                        )
                        reg_name = gr.Textbox(
                            label="Voice Persona Name",
                            placeholder="e.g. David Attenborough, Scarlett, Morgan...",
                            value="My Speaker"
                        )
                        reg_audio = gr.Audio(
                            label="Speaker Audio Sample (WAV / MP3 / M4A - Auto-Trimmed to <= 10s)",
                            type="filepath"
                        )
                        reg_chk_enhance = gr.Checkbox(
                            label="✨ Enhance & Denoise Voice Persona (Resemble Enhancer on GPU 1)",
                            value=True
                        )
                        gr.Markdown("ℹ️ *Ingested audio is automatically hard-trimmed to the first 10.0 seconds and transcribed automatically via Faster-Whisper Turbo v3 on GPU 0. Zero manual transcription required.*")

                        reg_btn = gr.Button("💾 Register Voice (.breeze)", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        reg_file = gr.File(label="Exported .breeze Voice File (Download & Share)")
                        reg_stats = gr.Markdown("Ready to register new voice persona.")

                reg_btn.click(
                    fn=register_voice_file,
                    inputs=[reg_name, reg_audio, reg_chk_enhance],
                    outputs=[reg_file, reg_stats, tts_voice_dropdown, clone_voice_dropdown, lf_voice_dropdown]
                )

            # ---------------- TAB 4: SPEECH-TO-SPEECH CONVERSION ----------------
            with gr.TabItem("🔄 Speech-to-Speech Voice Conversion (Mode 4)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("Transform one speaker's voice into another while preserving timing, cadence, and pronunciation.")
                        with gr.Row():
                            conv_src_audio = gr.Audio(label="Source Audio (Speech to convert)", type="filepath")
                            conv_ref_audio = gr.Audio(label="Target Voice Audio (Optional if selecting below)", type="filepath")

                        with gr.Row():
                            conv_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0],
                                label="Target Voice Persona (From Registry)",
                                scale=4
                            )
                            conv_refresh_btn = gr.Button("🔄", scale=1)

                        conv_refresh_btn.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[conv_voice_dropdown]
                        )

                        conv_ref_text = gr.Textbox(
                            label="Target Voice Transcript (Optional)",
                            placeholder="Transcript of the target reference voice...",
                            lines=1
                        )
                        with gr.Row():
                            conv_keep = gr.Slider(
                                0, 15, value=0, step=1,
                                label="Keep Acoustic Codebooks (0=Full Transfer, 1..3=Keep Pitch/Melody)"
                            )
                            conv_cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale (1.5 = Standard, 2.0 = Enhanced Conversion, 4.0 = Strong Steering)")
                            conv_seed = gr.Number(value=42, label="Random Seed")

                        conv_btn = gr.Button("▶ Convert Speech", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        conv_output_audio = gr.Audio(label="Converted Audio Output", type="filepath")
                        conv_stats = gr.Markdown("Ready for voice conversion.")

                conv_btn.click(
                    fn=voice_convert_generate,
                    inputs=[
                        conv_src_audio, conv_voice_dropdown, conv_ref_audio, conv_ref_text,
                        conv_keep, conv_cfg, conv_seed
                    ],
                    outputs=[conv_output_audio, conv_stats]
                )

            # ---------------- TAB 5: ASR & SUBTITLES TOOL ----------------
            with gr.TabItem("📝 ASR Transcription & Subtitles (.srt)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
                            ### Whisper ASR & Subtitle Generator (GPU 1)
                            Transcribes audio of **any duration** (from 5 seconds to 2 hours) with **constant ~288 MiB VRAM footprint**.
                            Uses Silero VAD to strip silence and yields timestamped SRT subtitles.
                            """
                        )
                        sub_audio = gr.Audio(label="Upload Audio File (Speech)", type="filepath")
                        sub_btn = gr.Button("🎙️ Generate Subtitles & Transcript", variant="primary")

                    with gr.Column(scale=3):
                        sub_text = gr.Textbox(label="Plain Transcript", lines=5)
                        sub_srt = gr.Textbox(label="SRT Subtitles (With Exact Timestamps)", lines=8)

                sub_btn.click(
                    fn=transcribe_subtitles_tool,
                    inputs=[sub_audio],
                    outputs=[sub_text, sub_srt]
                )

            # ---------------- TAB 6: TELEMETRY & ENGINE PROFILE ----------------
            with gr.TabItem("📊 Engine Configuration & Telemetry"):
                gr.Markdown(
                    f"""
                    ### Active Engine Profile (Zero Hardcoding)
                    * **Model GGUF**: `{APP_CONFIG['model_path']}`
                    * **Shared C++ Library**: `{APP_CONFIG['lib_path'] or 'Auto-detected'}`
                    * **Modular Q4 Surge**: `{'Enabled (Threshold: ' + str(APP_CONFIG['q4_threshold']) + ')' if APP_CONFIG['enable_q4_burst'] else 'Disabled (100% Precision)'}`
                    * **Vocoder Execution**: `{APP_CONFIG['vocoder_mode'].upper()} (32-Frame Stateful Causal Session)`
                    * **Voices Directory**: `{os.path.abspath(APP_CONFIG['voices_dir'])}`
                    * **ASR Microservice**: `{APP_CONFIG['asr_model']} pinned to CUDA:{APP_CONFIG['asr_device']}`
                    * **Engine Concurrency**: Autonomous Dual-Island Architecture (GPU 0: Island A, GPU 1: Island B)
                    """
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Breeze-TTS Studio Decoupled Web Application")
    parser.add_argument("--model", type=str, default="breeze-tts-2-q8_0.gguf", help="Path to base GGUF model")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so")
    parser.add_argument("--precision", type=str, choices=["int8", "fp16"], default="int8", help="Engine precision mode")
    parser.add_argument("--q4-model", type=str, default=None, help="Path to modular Q4 depth decoder GGUF")
    parser.add_argument("--enable-q4", action="store_true", help="Enable modular Q4 surge tiering")
    parser.add_argument("--q4-threshold", type=int, default=10, help="Active session threshold for Q4 burst")
    parser.add_argument("--vocoder-mode", type=str, choices=["on", "dynamic", "off"], default="on", help="Vocoder mode")
    parser.add_argument("--disable-enhancer", action="store_true", help="Disable Resemble Enhancer microservice")
    parser.add_argument("--disable-asr", action="store_true", help="Disable Whisper ASR microservice")
    parser.add_argument("--enhancer-chunk-sec", type=float, default=10.0, help="Sliding window chunk duration for Resemble Enhancer (10.0s saves 1.2 GB VRAM)")
    parser.add_argument("--enhancer-device", type=int, default=1, help="CUDA device index for Resemble Enhancer")
    parser.add_argument("--voices-dir", type=str, default="data/voices", help="Local directory for .breeze voice containers")
    parser.add_argument("--asr-device", type=int, default=0, help="CUDA device index for Whisper ASR")
    parser.add_argument("--asr-model", type=str, default="turbo", help="Whisper model size (default: turbo)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Web port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    # Populate Global Configuration
    APP_CONFIG["model_path"] = args.model
    APP_CONFIG["lib_path"] = args.lib
    APP_CONFIG["precision"] = args.precision
    APP_CONFIG["q4_model_path"] = args.q4_model
    APP_CONFIG["enable_q4_burst"] = args.enable_q4
    APP_CONFIG["q4_threshold"] = args.q4_threshold
    APP_CONFIG["vocoder_mode"] = args.vocoder_mode
    APP_CONFIG["enable_enhancer"] = not args.disable_enhancer
    APP_CONFIG["enable_asr"] = not args.disable_asr
    APP_CONFIG["enhancer_chunk_sec"] = args.enhancer_chunk_sec
    APP_CONFIG["enhancer_device"] = args.enhancer_device
    APP_CONFIG["voices_dir"] = args.voices_dir
    APP_CONFIG["asr_device"] = args.asr_device
    APP_CONFIG["asr_model"] = args.asr_model

    os.makedirs(APP_CONFIG["voices_dir"], exist_ok=True)
    os.makedirs(APP_CONFIG["outputs_dir"], exist_ok=True)

    print("⏳ Pre-warming cluster engine, orchestrator & micro-services...")
    try:
        orch = get_orchestrator()
        print(f"   ✅ EngineOrchestrator online & ready (Precision: {APP_CONFIG['precision'].upper()} | Enhancer: {APP_CONFIG['enable_enhancer']} | ASR: {APP_CONFIG['enable_asr']}).")
    except Exception as e:
        print(f"   ⚠️ Engine initialization deferred/failed: {e}")

    demo = build_app()
    print(f"\n🚀 Launching Breeze-TTS Studio on http://{args.host}:{args.port}")
    print(f"   • Model: {APP_CONFIG['model_path']} ({APP_CONFIG['precision'].upper()})")
    print(f"   • Voices Registry: {os.path.abspath(APP_CONFIG['voices_dir'])}")
    print(f"   • ASR: {APP_CONFIG['asr_model']} on CUDA:{APP_CONFIG['asr_device']}")
    print(f"   • Enhancer: CUDA:{APP_CONFIG['enhancer_device']} (Chunks: {APP_CONFIG['enhancer_chunk_sec']}s)")

    allowed_dirs = [
        os.path.abspath(APP_CONFIG["voices_dir"]),
        os.path.abspath(APP_CONFIG["outputs_dir"]),
        tempfile.gettempdir(),
        os.getcwd()
    ]
    if os.path.exists("/kaggle/working"):
        allowed_dirs.append(os.path.abspath("/kaggle/working"))
    if os.path.exists("/kaggle/working/data"):
        allowed_dirs.append(os.path.abspath("/kaggle/working/data"))
    if os.path.exists("/kaggle/working/data/outputs"):
        allowed_dirs.append(os.path.abspath("/kaggle/working/data/outputs"))
    allowed_dirs = list(set([d for d in allowed_dirs if os.path.exists(d)]))

    demo.queue(default_concurrency_limit=10)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_dirs
    )


if __name__ == "__main__":
    main()
