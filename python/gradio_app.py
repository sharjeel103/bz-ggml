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
    serialize_breeze_voice, deserialize_breeze_voice
)
from bz_ggml.asr import ASRService
from bz_ggml.audio_pipeline import (
    split_text_into_smart_chunks, stitch_audio_chunks,
    apply_resemble_enhancer, apply_studio_normalization,
    convert_to_mp3_256k, generate_1word_srt_file,
    PipelineJob, PipelineJobDispatcher, JobStage,
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
    "voices_dir": "data/voices",
    "outputs_dir": "/kaggle/working/data/outputs" if os.path.exists("/kaggle/working") else "data/outputs",
    "asr_device": 1,
    "asr_model": "base.en"
}

cluster_instance: Optional[DualInstanceCluster] = None
asr_service_instance: Optional[ASRService] = None
dispatcher_instance: Optional[PipelineJobDispatcher] = None


def get_dispatcher() -> PipelineJobDispatcher:
    """Initializes or returns the singleton PipelineJobDispatcher."""
    global dispatcher_instance
    if dispatcher_instance is None:
        out_dir = APP_CONFIG.get("outputs_dir", "data/outputs")
        os.makedirs(out_dir, exist_ok=True)
        dispatcher_instance = PipelineJobDispatcher(output_dir=out_dir)
    return dispatcher_instance


def get_cluster() -> DualInstanceCluster:
    """Initializes or returns the singleton DualInstanceCluster using APP_CONFIG."""
    global cluster_instance
    if cluster_instance is None:
        model_path = APP_CONFIG["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"[Gradio Studio] Initializing DualInstanceCluster with Model: {model_path}")
        print(f"  • Vocoder Mode: {APP_CONFIG['vocoder_mode']} | Q4 Burst: {APP_CONFIG['enable_q4_burst']}")
        cluster_instance = DualInstanceCluster(
            model_path=model_path,
            lib_path=APP_CONFIG["lib_path"],
            q4_model_path=APP_CONFIG["q4_model_path"],
            enable_q4_burst=APP_CONFIG["enable_q4_burst"],
            q4_threshold=APP_CONFIG["q4_threshold"],
            enable_vocoder=True,
            vocoder_mode=APP_CONFIG["vocoder_mode"]
        )
    return cluster_instance


def get_asr_service() -> ASRService:
    """Initializes or returns the singleton ASRService pinned to configured GPU."""
    global asr_service_instance
    if asr_service_instance is None:
        asr_service_instance = ASRService(
            model_size=APP_CONFIG["asr_model"],
            cuda_device=APP_CONFIG["asr_device"],
            compute_type="float16"
        )
    return asr_service_instance


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
    transcript: str
) -> Tuple[Optional[str], str, gr.Dropdown, gr.Dropdown]:
    """
    Encodes reference audio via engine's encode_voice_container() on GPU 0,
    receives in-memory bytes, saves to voices_dir, and updates UI dropdowns.
    """
    if not voice_name.strip():
        return None, "❌ Error: Voice name is required.", gr.update(), gr.update()
    if not audio_path or not os.path.exists(audio_path):
        return None, "❌ Error: Audio sample file is required.", gr.update(), gr.update()
    if not transcript.strip():
        return None, "❌ Error: Exact verbatim transcript is required.", gr.update(), gr.update()

    try:
        cluster = get_cluster()
        pcm = load_audio_24k(audio_path)
        t0 = time.time()
        # Engine contract: encode_voice_container returns in-memory .breeze bytes
        container_bytes = cluster.encode_voice_container(pcm, transcript.strip(), gpu_id=0)
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
            f"• **Acoustic Frames**: {meta['frames']} ({meta['frames'] * 0.08:.2f}s reference audio)\n"
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

    stream_pcm = (delivery_mode != "Studio Master (Full Single-Pass, Zero Clicks)")
    out_dir = tempfile.mkdtemp(prefix="breeze_tts_")

    try:
        cluster = get_cluster()
        task = UserTask(
            id=int(time.time() * 1000) % 100000,
            text=text,
            instruction=instruction,
            ref_text=ref_text,
            ref_codes=ref_codes,
            ref_frames=ref_frames,
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            max_steps=int(max_steps),
            stream_pcm=stream_pcm,
            client_has_vocoder=False
        )

        audio_queue = queue.Queue()

        def on_chunk(task_id: int, chunk_pcm: np.ndarray, is_eos: bool):
            if is_eos:
                audio_queue.put(("EOS", None, None))
            elif len(chunk_pcm) > 0:
                audio_queue.put(("CHUNK", chunk_pcm, None))

        def worker_thread():
            try:
                results = cluster.run_workload([task], out_dir=out_dir, on_pcm_chunk=on_chunk)
                res = results[0] if results else None
                audio_queue.put(("DONE", None, res))
            except Exception as e:
                audio_queue.put(("ERROR", None, e))

        t0 = time.time()
        w_thread = threading.Thread(target=worker_thread, daemon=True)
        w_thread.start()

        mode_label = "Studio Master (Full Single-Pass)" if not stream_pcm else "Real-Time Stateful Streaming (32-Frame Chunks)"
        voice_label = selected_voice if voice_path else "Default Native Speaker"

        total_samples = 0
        chunk_count = 0

        yield gr.skip(), None, gr.update(visible=False), f"⏳ Synthesizing audio on Dual GPU Islands ({mode_label})..."

        while True:
            try:
                msg_type, data, extra = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if not w_thread.is_alive():
                    break
                continue

            if msg_type == "CHUNK":
                chunk_pcm = data
                total_samples += len(chunk_pcm)
                chunk_count += 1
                audio_sec = total_samples / 24000.0
                elapsed = time.time() - t0
                pcm_i16 = (np.clip(chunk_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
                status_text = (
                    f"🔊 **Streaming Live Audio** | {mode_label}\n\n"
                    f"• **Voice Persona**: `{voice_label}`\n"
                    f"• **Live Audio Emitted**: {audio_sec:.2f}s ({total_samples:,} samples)\n"
                    f"• **Chunk #{chunk_count}**: {len(pcm_i16)} samples (32 frames @ 24 kHz)\n"
                    f"• **Elapsed Time**: {elapsed:.2f}s"
                )
                yield (24000, pcm_i16), gr.skip(), gr.skip(), status_text

            elif msg_type == "EOS":
                pass

            elif msg_type == "DONE":
                res = extra
                wall = time.time() - t0
                if res and res.wav_path and os.path.exists(res.wav_path):
                    master_wav = os.path.join(tempfile.gettempdir(), f"master_tts_{task.id}.wav")
                    shutil.copyfile(res.wav_path, master_wav)

                    stats_msg = (
                        f"✅ **Synthesis Complete** | {mode_label}\n\n"
                        f"• **Voice Persona**: `{voice_label}`\n"
                        f"• **Audio Length**: {res.audio_s:.2f}s ({int(res.audio_s * 24000):,} samples @ 24 kHz)\n"
                        f"• **Compute Time**: {res.compute_wall_s:.2f}s (Total Turnaround: {wall:.2f}s)\n"
                        f"• **Real-Time Speedup**: {res.rtf:.2f}x\n"
                        f"• **Worker Island**: {res.worker}\n"
                        f"• **Master File**: `{os.path.basename(master_wav)}` ({os.path.getsize(master_wav)/1024:.1f} KB)"
                    )
                    yield gr.skip(), master_wav, gr.update(value=master_wav, visible=True), stats_msg
                else:
                    yield gr.skip(), None, gr.update(visible=False), "❌ Error: Generation returned no audio."
                break

            elif msg_type == "ERROR":
                yield gr.skip(), None, gr.update(visible=False), f"❌ Error during synthesis: {str(extra)}"
                break

    except Exception as e:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Error during synthesis: {str(e)}"


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

    # Look up selected voice from pre-registered dropdown
    voice_path = get_voice_file_path(selected_voice)
    if not voice_path:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Error: Voice persona '{selected_voice}' not found in registry."
        return

    try:
        v_data = load_breeze_voice(voice_path)
        ref_codes = v_data["codes"]
        ref_frames = v_data["frames"]
        final_ref_text = v_data["text"]
        source_desc = f"Pre-registered Voice (`{selected_voice}`)"
    except Exception as e:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Error loading voice container: {str(e)}"
        return

    stream_pcm = (delivery_mode != "Studio Master (Full Single-Pass, Zero Clicks)")
    out_dir = tempfile.mkdtemp(prefix="breeze_clone_")

    try:
        cluster = get_cluster()
        task = UserTask(
            id=int(time.time() * 1000) % 100000,
            text=text,
            instruction=instruction,
            ref_text=final_ref_text,
            ref_codes=ref_codes,
            ref_frames=ref_frames,
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            max_steps=int(max_steps),
            stream_pcm=stream_pcm,
            client_has_vocoder=False
        )

        audio_queue = queue.Queue()

        def on_chunk(task_id: int, chunk_pcm: np.ndarray, is_eos: bool):
            if is_eos:
                audio_queue.put(("EOS", None, None))
            elif len(chunk_pcm) > 0:
                audio_queue.put(("CHUNK", chunk_pcm, None))

        def worker_thread():
            try:
                results = cluster.run_workload([task], out_dir=out_dir, on_pcm_chunk=on_chunk)
                res = results[0] if results else None
                audio_queue.put(("DONE", None, res))
            except Exception as e:
                audio_queue.put(("ERROR", None, e))

        t0 = time.time()
        w_thread = threading.Thread(target=worker_thread, daemon=True)
        w_thread.start()

        mode_label = "Studio Master (Full Single-Pass)" if not stream_pcm else "Real-Time Stateful Streaming (32-Frame Chunks)"
        total_samples = 0
        chunk_count = 0

        yield gr.skip(), None, gr.update(visible=False), f"⏳ Synthesizing cloned voice on Dual GPU Islands ({mode_label})..."

        while True:
            try:
                msg_type, data, extra = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if not w_thread.is_alive():
                    break
                continue

            if msg_type == "CHUNK":
                chunk_pcm = data
                total_samples += len(chunk_pcm)
                chunk_count += 1
                audio_sec = total_samples / 24000.0
                elapsed = time.time() - t0
                pcm_i16 = (np.clip(chunk_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
                status_text = (
                    f"🔊 **Streaming Live Cloned Audio** | {mode_label}\n\n"
                    f"• **Reference Voice**: {source_desc}\n"
                    f"• **Live Audio Emitted**: {audio_sec:.2f}s ({total_samples:,} samples)\n"
                    f"• **Chunk #{chunk_count}**: {len(pcm_i16)} samples (32 frames @ 24 kHz)\n"
                    f"• **Elapsed Time**: {elapsed:.2f}s"
                )
                yield (24000, pcm_i16), gr.skip(), gr.skip(), status_text

            elif msg_type == "EOS":
                pass

            elif msg_type == "DONE":
                res = extra
                wall = time.time() - t0
                if res and res.wav_path and os.path.exists(res.wav_path):
                    master_wav = os.path.join(tempfile.gettempdir(), f"master_clone_{task.id}.wav")
                    shutil.copyfile(res.wav_path, master_wav)

                    stats_msg = (
                        f"✅ **Cloned Successfully** | {mode_label}\n\n"
                        f"• **Reference Voice**: {source_desc}\n"
                        f"• **Audio Length**: {res.audio_s:.2f}s ({int(res.audio_s * 24000):,} samples @ 24 kHz)\n"
                        f"• **Compute Time**: {res.compute_wall_s:.2f}s (Total Turnaround: {wall:.2f}s)\n"
                        f"• **Real-Time Speedup**: {res.rtf:.2f}x\n"
                        f"• **Worker Island**: {res.worker}\n"
                        f"• **Master File**: `{os.path.basename(master_wav)}` ({os.path.getsize(master_wav)/1024:.1f} KB)"
                    )
                    yield gr.skip(), master_wav, gr.update(value=master_wav, visible=True), stats_msg
                else:
                    yield gr.skip(), None, gr.update(visible=False), "❌ Error: Voice cloning returned no audio."
                break

            elif msg_type == "ERROR":
                yield gr.skip(), None, gr.update(visible=False), f"❌ Error during voice cloning: {str(extra)}"
                break

    except Exception as e:
        yield gr.skip(), None, gr.update(visible=False), f"❌ Error during voice cloning: {str(e)}"


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
        yield "❌ Error: Script text is empty.", None, None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.skip()
        return

    dispatcher = get_dispatcher()
    cluster = get_cluster()
    asr = get_asr_service() if apply_srt else None

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
    job = PipelineJob(
        job_id=job_id,
        script_text=script_text.strip(),
        voice_name=selected_voice,
        instruction=instruction,
        cfg_scale=float(cfg_scale),
        temperature=float(temperature),
        seed=int(seed),
        pause_ms=int(pause_ms),
        target_words=int(target_words),
        max_words=int(max_words),
        apply_enhance=bool(apply_enhance),
        apply_master=bool(apply_master),
        apply_srt=bool(apply_srt)
    )

    update_queue = queue.Queue()

    def progress_callback(updated_job: PipelineJob):
        update_queue.put({
            "stage": updated_job.stage.value,
            "progress": updated_job.progress,
            "log": updated_job.status_log,
            "raw_mp3": updated_job.raw_mp3_path,
            "enhanced_mp3": updated_job.enhanced_mp3_path,
            "srt_path": updated_job.srt_path,
            "job": updated_job
        })

    def worker_thread():
        try:
            dispatcher.run_pipeline(
                job=job,
                cluster=cluster,
                asr_service=asr,
                ref_text=ref_text,
                ref_codes=ref_codes,
                ref_frames=ref_frames,
                update_callback=progress_callback
            )
            update_queue.put({"stage": "completed", "job": job})
        except Exception as e:
            update_queue.put({"stage": "failed", "error": str(e), "job": job})

    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()

    last_log = "🚀 Dispatching Long-Form Production Pipeline..."
    raw_mp3_val = None
    enh_mp3_val = None
    srt_text_val = ""

    while True:
        try:
            item = update_queue.get(timeout=0.25)
            stage = item.get("stage")

            if stage == "completed":
                done_job = item.get("job", job)
                saved_files = dispatcher.get_saved_files()
                final_srt_content = ""
                if done_job.srt_path and os.path.exists(done_job.srt_path):
                    with open(done_job.srt_path, "r", encoding="utf-8") as f:
                        final_srt_content = f.read()

                mins = int(done_job.total_audio_s // 60)
                secs = int(done_job.total_audio_s % 60)
                summary = (
                    f"🎉 **Production Pipeline Completed Successfully!**\n\n"
                    f"• **Job ID**: `{done_job.job_id}`\n"
                    f"• **Cohesive Chunks**: {done_job.chunks_completed} chunks synthesized concurrently across Dual GPUs\n"
                    f"• **Total Audio**: {done_job.total_audio_s:.2f}s ({mins}m {secs}s)\n"
                    f"• **Raw Voiceover**: `{os.path.basename(done_job.raw_mp3_path) if done_job.raw_mp3_path else 'N/A'}` (256k MP3)\n"
                    f"• **Mastered Voiceover**: `{os.path.basename(done_job.enhanced_mp3_path) if done_job.enhanced_mp3_path else 'N/A'}` (Broadcast Standard)\n"
                    f"• **Word Subtitles**: `{os.path.basename(done_job.srt_path) if done_job.srt_path else 'Disabled'}`"
                )
                yield (
                    summary,
                    done_job.raw_mp3_path,
                    done_job.enhanced_mp3_path,
                    final_srt_content,
                    gr.update(value=done_job.enhanced_mp3_path, visible=bool(done_job.enhanced_mp3_path)),
                    gr.update(value=done_job.raw_mp3_path, visible=bool(done_job.raw_mp3_path)),
                    gr.update(value=done_job.srt_path, visible=bool(done_job.srt_path and os.path.exists(done_job.srt_path))),
                    saved_files
                )
                break

            elif stage == "failed":
                err = item.get("error", "Unknown error")
                failed_job = item.get("job", job)
                yield (
                    f"❌ **Pipeline Note**: {err}",
                    failed_job.raw_mp3_path,
                    failed_job.enhanced_mp3_path,
                    "",
                    gr.update(value=failed_job.enhanced_mp3_path, visible=bool(failed_job.enhanced_mp3_path)),
                    gr.update(value=failed_job.raw_mp3_path, visible=bool(failed_job.raw_mp3_path)),
                    gr.update(visible=False),
                    dispatcher.get_saved_files()
                )
                break

            else:
                last_log = item.get("log", last_log)
                if item.get("raw_mp3"):
                    raw_mp3_val = item["raw_mp3"]
                if item.get("enhanced_mp3"):
                    enh_mp3_val = item["enhanced_mp3"]
                if item.get("srt_path") and os.path.exists(item["srt_path"]):
                    with open(item["srt_path"], "r", encoding="utf-8") as f:
                        srt_text_val = f.read()

                yield (
                    f"⏳ **Status**: {last_log}",
                    raw_mp3_val,
                    enh_mp3_val,
                    srt_text_val,
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
                            value=get_dispatcher().get_saved_files()
                        )
                        lf_history_refresh.click(
                            fn=lambda: get_dispatcher().get_saved_files(),
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
                            label="Speaker Audio Sample (WAV / MP3, max 10s)",
                            type="filepath"
                        )
                        with gr.Row():
                            reg_asr_btn = gr.Button("🎙️ Auto-Transcribe Sample (GPU 1 Whisper ASR)", variant="secondary")
                        reg_transcript = gr.Textbox(
                            label="Verbatim Transcript (Human in the Loop Review & Edit)",
                            placeholder="Transcript will appear here. Edit for any mistakes...",
                            lines=3
                        )
                        reg_asr_btn.click(
                            fn=transcribe_audio_sample,
                            inputs=[reg_audio],
                            outputs=[reg_transcript]
                        )

                        reg_btn = gr.Button("💾 Register Voice (.breeze)", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        reg_file = gr.File(label="Exported .breeze Voice File (Download & Share)")
                        reg_stats = gr.Markdown("Ready to register new voice persona.")

                reg_btn.click(
                    fn=register_voice_file,
                    inputs=[reg_name, reg_audio, reg_transcript],
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
    parser.add_argument("--q4-model", type=str, default=None, help="Path to modular Q4 depth decoder GGUF")
    parser.add_argument("--enable-q4", action="store_true", help="Enable modular Q4 surge tiering")
    parser.add_argument("--q4-threshold", type=int, default=10, help="Active session threshold for Q4 burst")
    parser.add_argument("--vocoder-mode", type=str, choices=["on", "dynamic", "off"], default="on", help="Vocoder mode")
    parser.add_argument("--voices-dir", type=str, default="data/voices", help="Local directory for .breeze voice containers")
    parser.add_argument("--asr-device", type=int, default=1, help="CUDA device index for Whisper ASR")
    parser.add_argument("--asr-model", type=str, default="small", help="Whisper model size")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Web port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    # Populate Global Configuration
    APP_CONFIG["model_path"] = args.model
    APP_CONFIG["lib_path"] = args.lib
    APP_CONFIG["q4_model_path"] = args.q4_model
    APP_CONFIG["enable_q4_burst"] = args.enable_q4
    APP_CONFIG["q4_threshold"] = args.q4_threshold
    APP_CONFIG["vocoder_mode"] = args.vocoder_mode
    APP_CONFIG["voices_dir"] = args.voices_dir
    APP_CONFIG["asr_device"] = args.asr_device
    APP_CONFIG["asr_model"] = args.asr_model

    os.makedirs(APP_CONFIG["voices_dir"], exist_ok=True)
    os.makedirs(APP_CONFIG["outputs_dir"], exist_ok=True)

    print("⏳ Pre-warming cluster engine & ASR service...")
    try:
        get_cluster()
        print("   ✅ DualInstanceCluster initialized & ready.")
    except Exception as e:
        print(f"   ⚠️ Cluster initialization deferred/failed: {e}")

    try:
        get_asr_service()
        print("   ✅ Whisper ASR initialized & ready.")
    except Exception as e:
        print(f"   ⚠️ ASR initialization deferred/failed: {e}")

    demo = build_app()
    print(f"\n🚀 Launching Breeze-TTS Studio on http://{args.host}:{args.port}")
    print(f"   • Model: {APP_CONFIG['model_path']}")
    print(f"   • Voices Registry: {os.path.abspath(APP_CONFIG['voices_dir'])}")
    print(f"   • ASR Device: CUDA:{APP_CONFIG['asr_device']}")

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
