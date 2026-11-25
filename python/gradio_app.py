#!/usr/bin/env python3
"""
Breeze-TTS Studio: Decoupled Gradio Web Application Client
Provides in-browser Long-Form Production Studio, Text-to-Speech, Zero-Shot Voice Cloning,
Auto-Gated Voice Registration (.breeze), Speech-to-Speech Conversion (Mode 4), and Whisper Turbo Subtitles.
Consumes public bz_ggml APIs directly with verified Option A timbre-lock defaults.
"""

import argparse
import os
import queue
import sys
import tempfile
import threading
import time
from typing import Optional, Tuple, List, Dict, Any, Generator
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bz_ggml import (
    DualInstanceCluster,
    VoiceConversionTask,
    load_breeze_voice,
    deserialize_breeze_voice,
    EngineOrchestrator,
    EngineCapabilities,
    Parcel,
    ParcelResult,
    save_pcm_to_mp3,
    save_pcm_to_wav,
)
from bz_ggml.asr import ASRService
from bz_ggml.audio_pipeline import format_chunk_preview_markdown

import gradio as gr

DEFAULT_OPTION_A_INSTRUCTION = "Speak clearly and naturally."
REG_ENHANCE_CHOICES = [
    "Auto (Skip Resemble if Clean >= 30 dB)",
    "Always Enhance (44.1 kHz Denoise)",
    "Off (Gentle Mastering Only)",
]

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
        resolved_prec = DualInstanceCluster.resolve_precision(model_path, APP_CONFIG.get("precision"))
        APP_CONFIG["precision"] = resolved_prec
        print(f"[Gradio Studio] Initializing DualInstanceCluster with Model: {model_path}")
        print(f"  - Precision: {resolved_prec.upper()} | Vocoder Mode: {APP_CONFIG['vocoder_mode']} | Q4 Burst: {APP_CONFIG['enable_q4_burst']}")
        cluster_instance = DualInstanceCluster(
            model_path=model_path,
            lib_path=APP_CONFIG["lib_path"],
            q4_model_path=APP_CONFIG["q4_model_path"],
            enable_q4_burst=APP_CONFIG["enable_q4_burst"],
            q4_threshold=APP_CONFIG["q4_threshold"],
            enable_vocoder=True,
            vocoder_mode=APP_CONFIG["vocoder_mode"],
            precision=resolved_prec,
            enable_enhancer=APP_CONFIG.get("enable_enhancer", True),
            enable_asr=APP_CONFIG.get("enable_asr", True),
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
            compute_type="float16",
        )
    return asr_service_instance


def get_orchestrator() -> EngineOrchestrator:
    """Initializes or returns the singleton EngineOrchestrator."""
    global orchestrator_instance
    if orchestrator_instance is None:
        cluster = get_cluster()
        asr = get_asr_service()
        caps = EngineCapabilities(
            precision=cluster.precision,
            enable_q4_surge=APP_CONFIG.get("enable_q4_burst", False),
            vocoder_mode=APP_CONFIG.get("vocoder_mode", "on"),
            enable_enhancer=APP_CONFIG.get("enable_enhancer", True),
            enable_asr=APP_CONFIG.get("enable_asr", True),
            chunk_seconds_enhancer=APP_CONFIG.get("enhancer_chunk_sec", 10.0),
            asr_model_size=APP_CONFIG.get("asr_model", "turbo"),
            asr_device_index=APP_CONFIG.get("asr_device", 0),
            enhancer_device_index=APP_CONFIG.get("enhancer_device", 1),
        )
        orchestrator_instance = EngineOrchestrator(
            cluster=cluster,
            asr_service=asr,
            capabilities=caps,
        )
    return orchestrator_instance


def load_audio_24k(audio_path: str) -> List[float]:
    """Loads an audio file at 24 kHz mono float32 headroom via the core pipeline loader."""
    from bz_ggml.audio_pipeline import load_any_audio
    pcm, _ = load_any_audio(audio_path, target_sr=24000)
    return pcm.astype(np.float32).tolist()


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


def _load_selected_voice(selected_voice: str) -> Tuple[Optional[List[int]], int, Optional[str], Optional[str]]:
    """Loads reference tokens, frame count, transcript, and path when a saved .breeze persona is selected."""
    voice_path = get_voice_file_path(selected_voice)
    if not voice_path:
        return None, 0, None, None
    v_data = load_breeze_voice(voice_path)
    return v_data["codes"], int(v_data["frames"]), v_data["text"], voice_path


def _resolve_reg_enhance_mode(enhance_mode: Any) -> Tuple[Any, str]:
    """Maps Radio label or boolean into Parcel.voice_registration enhance argument."""
    if isinstance(enhance_mode, bool):
        return enhance_mode, ("Resemble 44.1 kHz" if enhance_mode else "Direct Gentle Master")
    text = str(enhance_mode or "").strip().lower()
    if text.startswith("auto"):
        return "auto", "Auto SNR Gate (Resemble skipped if >= 30 dB + Gentle Master)"
    if text.startswith("off") or text in ("false", "0", "none"):
        return False, "Off (Gentle Master Only)"
    return True, "Always Enhance (Resemble 44.1 kHz + Gentle Master)"


def transcribe_subtitles_tool(audio_path: Optional[str]) -> Tuple[str, str]:
    """Transcribes audio of arbitrary length into full text and SRT timestamped subtitles."""
    if not audio_path or not os.path.exists(audio_path):
        return "Error: Please upload an audio file.", ""
    try:
        asr = get_asr_service()
        if asr is None:
            return "ASR Microservice is disabled on this instance (--disable-asr).", ""
        transcript, srt_content, _ = asr.transcribe(audio_path, vad_filter=True, return_subtitles=True)
        return transcript, srt_content
    except Exception as e:
        return f"ASR Error: {str(e)}", ""


def register_voice_file(
    voice_name: str,
    audio_path: Optional[str],
    enhance_mode: Any = REG_ENHANCE_CHOICES[0],
    transcript_override: str = "",
) -> Tuple[Optional[str], str, Any, Any, Any, Any]:
    """
    Submits a Voice Persona Registration parcel [ENHANCE (auto/on/off), MASTER, ASR (unless transcript supplied), VOICE_REGISTER]
    to EngineOrchestrator, saves .breeze container, and refreshes all voice dropdowns.
    """
    if not voice_name.strip():
        return None, "Error: Voice name is required.", gr.update(), gr.update(), gr.update(), gr.update()
    if not audio_path or not os.path.exists(audio_path):
        return None, "Error: Audio sample file is required.", gr.update(), gr.update(), gr.update(), gr.update()

    try:
        orchestrator = get_orchestrator()
        enhance_arg, enhance_desc = _resolve_reg_enhance_mode(enhance_mode)
        clean_tx = transcript_override.strip() or None
        parcel = Parcel.voice_registration(
            audio_input=audio_path,
            voice_name=voice_name.strip(),
            transcript=clean_tx,
            enhance=enhance_arg,
            master=True,
            max_duration_s=10.0,
        )
        t0 = time.time()
        res = orchestrator.dispatch(parcel)
        if not res.success or res.voice_bytes is None:
            raise RuntimeError(res.error_message or "Failed to encode voice container")

        container_bytes = res.voice_bytes
        enc_time = time.time() - t0

        v_dir = APP_CONFIG["voices_dir"]
        os.makedirs(v_dir, exist_ok=True)
        safe_name = "".join(c for c in voice_name.strip() if c.isalnum() or c in (" ", "_", "-")).strip()
        breeze_filename = f"{safe_name.lower().replace(' ', '_')}.breeze"
        breeze_path = os.path.join(v_dir, breeze_filename)

        with open(breeze_path, "wb") as f:
            f.write(container_bytes)

        temp_export = os.path.join(tempfile.gettempdir(), breeze_filename)
        with open(temp_export, "wb") as f:
            f.write(container_bytes)

        meta = deserialize_breeze_voice(container_bytes)
        file_kb = len(container_bytes) / 1024.0
        tx_source = "Manual Override" if clean_tx else f"Auto Whisper ({APP_CONFIG['asr_model']})"

        msg = (
            f"### Voice Registered Successfully\n\n"
            f"- **Voice Persona**: `{voice_name.strip()}`\n"
            f"- **Transcript ({tx_source})**: \"{meta['text']}\"\n"
            f"- **Conditioning**: {enhance_desc}\n"
            f"- **Acoustic Frames**: {meta['frames']} ({meta['frames'] * 0.08:.2f}s @ 12.5 Hz, capped <= 10s)\n"
            f"- **Discrete Tokens**: {len(meta['codes'])} tokens (16 codebooks)\n"
            f"- **Total Encode Time**: {enc_time:.2f}s (Timings: {res.stage_timings})\n"
            f"- **Saved Registry File**: `{breeze_path}` ({file_kb:.1f} KB)"
        )

        updated_choices = list_saved_voices()
        selected_title = breeze_filename[:-7].replace("_", " ").title()
        dd_upd = gr.update(choices=updated_choices, value=selected_title)
        return temp_export, msg, dd_upd, dd_upd, dd_upd, dd_upd
    except Exception as e:
        return None, f"Voice Registration Failed: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update()


def _run_single_tts_ui(
    text: str,
    instruction: str,
    selected_voice: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    depth_temperature: float,
    depth_cfg_codebooks: int,
    seed: int,
    require_voice: bool = False,
) -> Generator[Tuple[Any, Any, str], None, None]:
    """Shared handler for Tab 1 (TTS & Voice Design) and Tab 2 (Zero-Shot Cloning)."""
    if not text.strip():
        yield None, gr.update(visible=False), "Error: Text input is empty."
        return

    try:
        ref_codes, ref_frames, ref_text, voice_path = _load_selected_voice(selected_voice)
    except Exception as e:
        yield None, gr.update(visible=False), f"Error loading voice container: {str(e)}"
        return

    if require_voice and not voice_path:
        yield None, gr.update(visible=False), f"Error: Voice persona '{selected_voice}' not found in registry."
        return

    voice_label = selected_voice if voice_path else "Default Native Speaker"
    orchestrator = get_orchestrator()
    prefix = "clone" if require_voice else "tts"

    parcel = Parcel.tts_pipeline(
        script_text=text.strip(),
        voice_name=selected_voice,
        instruction=(instruction or DEFAULT_OPTION_A_INSTRUCTION).strip(),
        cfg_scale=float(cfg_scale),
        follower_cfg_scale=float(cfg_scale),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        depth_temperature=float(depth_temperature),
        depth_cfg_codebooks=int(depth_cfg_codebooks),
        seed=int(seed),
        enhance=False,
        master=True,
        srt=False,
        ref_codes=ref_codes,
        ref_frames=ref_frames,
        ref_text=ref_text,
        parcel_id=f"{prefix}_{int(time.time() * 1000)}",
    )

    yield None, gr.update(visible=False), f"Synthesizing on Dual GPUs ({voice_label})..."

    t0 = time.time()
    res = orchestrator.dispatch(parcel)
    wall = time.time() - t0

    if not res.success or res.audio_pcm is None:
        yield None, gr.update(visible=False), f"Synthesis failed: {res.error_message}"
        return

    out_wav = os.path.join(tempfile.gettempdir(), f"{parcel.parcel_id}.wav")
    save_pcm_to_wav(res.audio_pcm, res.sample_rate, out_wav)
    rt_speed = res.duration_s / max(0.01, wall)

    stats_msg = (
        f"### Synthesis Complete (Studio Master)\n\n"
        f"- **Voice Persona**: `{voice_label}`\n"
        f"- **Chunks**: {res.chunks_count} | **Audio Length**: {res.duration_s:.2f}s @ {res.sample_rate} Hz\n"
        f"- **Compute Time**: {wall:.2f}s (**{rt_speed:.2f}x** realtime)\n"
        f"- **Sampling**: CFG={float(cfg_scale):.1f}, Temp={float(temperature):.2f}, DepthTemp={float(depth_temperature):.2f}, DepthCFG={int(depth_cfg_codebooks)}\n"
        f"- **Output File**: `{os.path.basename(out_wav)}` ({os.path.getsize(out_wav)/1024:.1f} KB)"
    )
    yield out_wav, gr.update(value=out_wav, visible=True), stats_msg


def tts_generate(
    text: str,
    instruction: str,
    selected_voice: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    depth_temperature: float,
    depth_cfg_codebooks: int,
    seed: int,
) -> Generator[Tuple[Any, Any, str], None, None]:
    yield from _run_single_tts_ui(
        text, instruction, selected_voice,
        cfg_scale, temperature, top_p, top_k, repetition_penalty,
        depth_temperature, depth_cfg_codebooks, seed,
        require_voice=False,
    )


def voice_clone_generate(
    text: str,
    selected_voice: str,
    instruction: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    depth_temperature: float,
    depth_cfg_codebooks: int,
    seed: int,
) -> Generator[Tuple[Any, Any, str], None, None]:
    yield from _run_single_tts_ui(
        text, instruction, selected_voice,
        cfg_scale, temperature, top_p, top_k, repetition_penalty,
        depth_temperature, depth_cfg_codebooks, seed,
        require_voice=True,
    )


def voice_convert_generate(
    source_audio: Optional[str],
    selected_voice: str,
    ref_audio: Optional[str],
    ref_text: str,
    keep_acoustic: int,
    cfg_scale: float,
    seed: int,
) -> Tuple[Optional[str], str]:
    if not source_audio:
        return None, "Error: Source audio file is required."

    ref_codes, ref_frames, final_ref_text, voice_path = _load_selected_voice(selected_voice)
    if not voice_path:
        if ref_audio and os.path.exists(ref_audio):
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
            return None, "Error: Could not encode source audio frames."

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
            stream_pcm=False,
        )

        audio_samples, _, stats = cluster.convert_voice_task(task)
        if not audio_samples:
            return None, "Error: Voice conversion produced no samples."

        out_wav = os.path.join(tempfile.gettempdir(), f"converted_voice_{task.id}.wav")
        save_pcm_to_wav(np.asarray(audio_samples, dtype=np.float32), 24000, out_wav)

        target_desc = selected_voice if voice_path else (os.path.basename(ref_audio) if ref_audio else "Reference Audio")
        stats_msg = (
            f"### Voice Conversion Complete (Mode 4)\n\n"
            f"- **Source Audio**: `{os.path.basename(source_audio)}` ({src_frames * 0.08:.2f}s, {src_frames} frames)\n"
            f"- **Target Persona**: `{target_desc}`\n"
            f"- **Wall Time**: {stats['compute_wall_s']:.2f}s (RTF: {stats['rtf']:.3f}x)\n"
            f"- **Acoustic Codebooks Kept**: {keep_acoustic}/15\n"
            f"- **Worker Island**: {stats['worker']}"
        )
        return out_wav, stats_msg
    except Exception as e:
        return None, f"Error during voice conversion: {str(e)}"


def preview_chunks_ui(text: str, selected_voice: str, auto_chunk_bounds: bool, target_w: int, max_w: int) -> str:
    """Computes live semantic chunk preview matching orchestrator auto-sizing for cloned vs uncloned."""
    if auto_chunk_bounds:
        has_voice = bool(get_voice_file_path(selected_voice))
        eff_tw, eff_mw = (45, 60) if has_voice else (110, 140)
    else:
        eff_tw, eff_mw = int(target_w), int(max_w)
    return format_chunk_preview_markdown(text, target_words=eff_tw, max_words=eff_mw)


def run_longform_pipeline_ui(
    script_text: str,
    selected_voice: str,
    instruction: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    depth_temperature: float,
    depth_cfg_codebooks: int,
    seed: int,
    pause_ms: int,
    auto_chunk_bounds: bool,
    target_words: int,
    max_words: int,
    apply_enhance: bool,
    apply_master: bool,
    apply_srt: bool,
) -> Generator[Tuple[str, Optional[str], Optional[str], Optional[str], Any, Any, Any, Any], None, None]:
    if not script_text.strip():
        yield "Error: Script text is empty.", None, None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), list_deliverable_files()
        return

    out_dir = APP_CONFIG["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)

    try:
        ref_codes, ref_frames, ref_text, voice_path = _load_selected_voice(selected_voice)
    except Exception as e:
        yield f"Error loading registered voice: {e}", None, None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), list_deliverable_files()
        return

    eff_target_words = None if auto_chunk_bounds else int(target_words)
    eff_max_words = None if auto_chunk_bounds else int(max_words)
    eff_enhance = bool(apply_enhance and APP_CONFIG.get("enable_enhancer", True))
    eff_srt = bool(apply_srt and APP_CONFIG.get("enable_asr", True))

    job_id = f"prod_{int(time.time() * 1000)}"
    parcel = Parcel.tts_pipeline(
        script_text=script_text.strip(),
        voice_name=selected_voice,
        instruction=(instruction or DEFAULT_OPTION_A_INSTRUCTION).strip(),
        cfg_scale=float(cfg_scale),
        follower_cfg_scale=float(cfg_scale),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        depth_temperature=float(depth_temperature),
        depth_cfg_codebooks=int(depth_cfg_codebooks),
        seed=int(seed),
        pause_ms=int(pause_ms),
        target_words=eff_target_words,
        max_words=eff_max_words,
        enhance=eff_enhance,
        master=bool(apply_master),
        srt=eff_srt,
        ref_codes=ref_codes,
        ref_frames=ref_frames,
        ref_text=ref_text,
        parcel_id=job_id,
    )

    update_queue: queue.Queue = queue.Queue()

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

    last_log = "Dispatching Parcel to Engine Orchestrator..."
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
                        f"Engine Error: {res.error_message}",
                        None, None, "",
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        list_deliverable_files(),
                    )
                    break

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
                    f"### Parcel Pipeline Completed Successfully\n\n"
                    f"- **Parcel ID**: `{res.parcel_id}` ({'Cloned: ' + selected_voice if voice_path else 'Default Voice'})\n"
                    f"- **Cohesive Chunks**: {res.chunks_count} chunks synthesized cleanly on Dual GPUs\n"
                    f"- **Total Audio**: {res.duration_s:.2f}s ({mins}m {secs}s @ {res.sample_rate} Hz)\n"
                    f"- **Raw Voiceover**: `{os.path.basename(raw_mp3_path) if raw_mp3_path else 'N/A'}` (256k MP3)\n"
                    f"- **Mastered Voiceover**: `{os.path.basename(mastered_mp3_path) if mastered_mp3_path else 'N/A'}` (Option A Broadcast)\n"
                    f"- **Word Subtitles**: `{os.path.basename(srt_file_path) if srt_file_path else 'Disabled'}`\n"
                    f"- **Stage Timings**: {', '.join(f'{k}: {v:.2f}s' for k, v in res.stage_timings.items())}"
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
                    saved_files,
                )
                break

            elif stage == "failed":
                err = item.get("error", "Unknown error")
                yield (
                    f"Pipeline Failure: {err}",
                    raw_mp3_path,
                    mastered_mp3_path,
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    list_deliverable_files(),
                )
                break

            else:
                last_log = item.get("msg", last_log)
                pct = item.get("progress", 0.0)
                yield (
                    f"[{stage.upper()}] ({int(pct*100)}%): {last_log}",
                    raw_mp3_path,
                    mastered_mp3_path,
                    srt_content,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.skip(),
                )

        except queue.Empty:
            if not thread.is_alive():
                break


def _build_sampling_controls(accordion_open: bool = False) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Creates shared Option A verified sampling controls for Long-Form, TTS, and Cloning tabs."""
    with gr.Row():
        cfg = gr.Slider(1.0, 5.0, value=1.6, step=0.1, label="CFG Scale (1.6 = Option A Cloned Lock)")
        temp = gr.Slider(0.05, 1.2, value=0.55, step=0.05, label="Temperature (0.55 = Balanced)")
        seed = gr.Number(value=42, label="Random Seed", precision=0)

    with gr.Accordion("Advanced Option A Sampling & Depth Timbre Lock", open=accordion_open):
        with gr.Row():
            top_p = gr.Slider(0.1, 1.0, value=0.85, step=0.01, label="Top-P Nucleus")
            top_k = gr.Slider(1, 100, value=30, step=1, label="Top-K")
            rep_pen = gr.Slider(1.0, 1.4, value=1.08, step=0.01, label="Repetition Penalty (1.08)")
        with gr.Row():
            depth_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Depth Temperature (0.0 = Clean Argmax)")
            depth_cfg = gr.Slider(0, 15, value=3, step=1, label="Depth CFG Codebooks (3 = Timbre-Lock)")

    return cfg, temp, top_p, top_k, rep_pen, depth_temp, depth_cfg, seed


def _is_gradio_6_plus() -> bool:
    try:
        return int(str(getattr(gr, "__version__", "4")).split(".")[0]) >= 6
    except Exception:
        return False


def build_app() -> gr.Blocks:
    custom_css = """
    .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    """
    initial_voices = list_saved_voices()
    blocks_kwargs: Dict[str, Any] = {"title": "Breeze-TTS Studio"}
    if not _is_gradio_6_plus():
        blocks_kwargs["css"] = custom_css
        blocks_kwargs["theme"] = gr.themes.Soft()

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            """
            # Breeze-TTS Studio & Voice Lab
            ### Option A Locked-Consistency Speech Synthesis, Auto-Gated Cloning & Subtitles
            *Step-480 Tiled Vocoder | Isolated Chunk Decoding + RMS Leveling | Whisper Turbo ASR | Portable `.breeze` Containers*
            """
        )

        with gr.Tabs():
            # ---------------- TAB 0: LONG-FORM PRODUCTION PIPELINE ----------------
            with gr.TabItem("Long-Form Studio (Parallel Chunks & Mastering)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        lf_script = gr.Textbox(
                            label="Full Script to Synthesize (Unlimited Length)",
                            placeholder="Paste your long-form script here...",
                            lines=10,
                            value=(
                                "On a humid September morning in nineteen seventy-seven, a Titan Centaur rocket lifted off from Cape Canaveral, "
                                "piercing the morning haze with a sustained pillar of flame.\n\n"
                                "Fastened to its upper stage was Voyager One, carrying within its delicate instruments the grandest aspirations "
                                "of the human species into the cosmic dark."
                            ),
                        )
                        with gr.Row():
                            lf_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0],
                                label="Voice Persona (From Voices Registry)",
                                scale=4,
                            )
                            lf_refresh_voice = gr.Button("Refresh", scale=1)

                        lf_refresh_voice.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[lf_voice_dropdown],
                        )

                        lf_instruction = gr.Textbox(
                            label="Speaker Style / Narration Direction",
                            value=DEFAULT_OPTION_A_INSTRUCTION,
                        )

                        lf_cfg, lf_temp, lf_top_p, lf_top_k, lf_rep, lf_dtemp, lf_dcfg, lf_seed = _build_sampling_controls(False)
                        lf_pause = gr.Slider(0, 600, value=200, step=25, label="Inter-Chunk Breath Pause (ms)")

                        with gr.Accordion("Semantic Chunking Controls & Preview", open=False):
                            lf_auto_chunks = gr.Checkbox(
                                value=True,
                                label="Auto Chunk Sizing (Recommended: 45w/60w Cloned, 110w/140w Uncloned)",
                            )
                            with gr.Row():
                                lf_target_words = gr.Slider(25, 140, value=45, step=5, label="Manual Target Words/Chunk")
                                lf_max_words = gr.Slider(40, 170, value=60, step=5, label="Manual Max Words/Chunk")
                            lf_preview_btn = gr.Button("Preview Chunks & Word Breakdown")
                            lf_preview_box = gr.Markdown("Click **Preview Chunks** to inspect partitions.")

                        lf_preview_btn.click(
                            fn=preview_chunks_ui,
                            inputs=[lf_script, lf_voice_dropdown, lf_auto_chunks, lf_target_words, lf_max_words],
                            outputs=[lf_preview_box],
                        )

                        with gr.Row():
                            lf_chk_enhance = gr.Checkbox(
                                value=False,
                                label="Apply Resemble Enhancer (44.1 kHz Denoise - keep off for raw timbre)",
                            )
                            lf_chk_master = gr.Checkbox(value=True, label="Apply Studio Mastering (optionA_rich_bass)")
                            lf_chk_srt = gr.Checkbox(
                                value=APP_CONFIG.get("enable_asr", True),
                                label="Generate 1-Word SRT Subtitles",
                            )

                        lf_submit_btn = gr.Button("Dispatch Long-Form Pipeline", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        lf_status_box = gr.Markdown("Ready to process script.")
                        lf_audio_raw = gr.Audio(label="Raw Stitched Voiceover (256k MP3)", type="filepath")
                        lf_audio_mastered = gr.Audio(label="Mastered Voiceover (256k MP3)", type="filepath")
                        lf_srt_box = gr.Textbox(label="1-Word SRT Subtitles Preview", lines=7)

                        with gr.Row():
                            lf_down_mastered = gr.DownloadButton("Download Mastered MP3", visible=False, variant="primary")
                            lf_down_raw = gr.DownloadButton("Download Raw MP3", visible=False)
                            lf_down_srt = gr.DownloadButton("Download 1-Word SRT", visible=False)

                        lf_history_refresh = gr.Button("Refresh Saved Files")
                        lf_history_files = gr.File(
                            label="Saved Output Files (.mp3 / .srt / .wav)",
                            file_count="multiple",
                            value=list_deliverable_files(),
                        )
                        lf_history_refresh.click(
                            fn=list_deliverable_files,
                            inputs=[],
                            outputs=[lf_history_files],
                        )

                lf_submit_btn.click(
                    fn=run_longform_pipeline_ui,
                    inputs=[
                        lf_script, lf_voice_dropdown, lf_instruction,
                        lf_cfg, lf_temp, lf_top_p, lf_top_k, lf_rep, lf_dtemp, lf_dcfg,
                        lf_seed, lf_pause, lf_auto_chunks, lf_target_words, lf_max_words,
                        lf_chk_enhance, lf_chk_master, lf_chk_srt,
                    ],
                    outputs=[
                        lf_status_box, lf_audio_raw, lf_audio_mastered, lf_srt_box,
                        lf_down_mastered, lf_down_raw, lf_down_srt, lf_history_files,
                    ],
                )

            # ---------------- TAB 1: TEXT-TO-SPEECH & VOICE DESIGN ----------------
            with gr.TabItem("Text-to-Speech & Voice Design"):
                with gr.Row():
                    with gr.Column(scale=3):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            value="Welcome to Breeze-TTS Studio. Synthesizing high-fidelity speech with locked timbre across dual GPUs.",
                        )
                        tts_instruction = gr.Textbox(
                            label="Speaker Persona / Style Instruction",
                            value=DEFAULT_OPTION_A_INSTRUCTION,
                        )
                        with gr.Row():
                            tts_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0],
                                label="Voice Persona (Optional Registry Selection)",
                                scale=4,
                            )
                            tts_refresh_btn = gr.Button("Refresh", scale=1)

                        tts_refresh_btn.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[tts_voice_dropdown],
                        )

                        tts_cfg, tts_temp, tts_top_p, tts_top_k, tts_rep, tts_dtemp, tts_dcfg, tts_seed = _build_sampling_controls(False)
                        tts_btn = gr.Button("Generate Audio", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        tts_master_audio = gr.Audio(label="Synthesized Audio (Mastered WAV)", type="filepath")
                        tts_download_file = gr.File(label="Download Master WAV", file_types=[".wav"], visible=False)
                        tts_stats = gr.Markdown("Ready to synthesize.")

                tts_btn.click(
                    fn=tts_generate,
                    inputs=[
                        tts_text, tts_instruction, tts_voice_dropdown,
                        tts_cfg, tts_temp, tts_top_p, tts_top_k, tts_rep, tts_dtemp, tts_dcfg, tts_seed,
                    ],
                    outputs=[tts_master_audio, tts_download_file, tts_stats],
                )

            # ---------------- TAB 2: ZERO-SHOT VOICE CLONING ----------------
            with gr.TabItem("Zero-Shot Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=3):
                        clone_text = gr.Textbox(
                            label="Text to Speak in Cloned Voice",
                            lines=4,
                            value="This is a demonstration of zero-shot voice cloning using a pre-registered .breeze voice container.",
                        )
                        with gr.Row():
                            clone_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0] if initial_voices else None,
                                label="Registered Voice Persona",
                                scale=4,
                            )
                            clone_refresh_btn = gr.Button("Refresh", scale=1)

                        clone_refresh_btn.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[clone_voice_dropdown],
                        )

                        clone_instruction = gr.Textbox(
                            label="Delivery Direction (Neutral Recommended for Cloning)",
                            value=DEFAULT_OPTION_A_INSTRUCTION,
                        )
                        cl_cfg, cl_temp, cl_top_p, cl_top_k, cl_rep, cl_dtemp, cl_dcfg, cl_seed = _build_sampling_controls(False)
                        clone_btn = gr.Button("Clone Voice & Synthesize", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        clone_master_audio = gr.Audio(label="Cloned Audio Output (Mastered WAV)", type="filepath")
                        clone_download_file = gr.File(label="Download Cloned WAV", file_types=[".wav"], visible=False)
                        clone_stats = gr.Markdown("Select a registered voice to begin.")

                clone_btn.click(
                    fn=voice_clone_generate,
                    inputs=[
                        clone_text, clone_voice_dropdown, clone_instruction,
                        cl_cfg, cl_temp, cl_top_p, cl_top_k, cl_rep, cl_dtemp, cl_dcfg, cl_seed,
                    ],
                    outputs=[clone_master_audio, clone_download_file, clone_stats],
                )

            # ---------------- TAB 3: VOICE REGISTRATION & MANAGEMENT ----------------
            with gr.TabItem("Voice Registration (.breeze)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
                            ### Voice Persona Registration
                            1. Upload reference speech (clean audio <= 10.0s is optimal; longer clips auto-trim to 10.0s).
                            2. **Auto SNR Gate** skips Resemble resynthesis automatically when input SNR >= 30 dB, preserving original timbre, and runs GPU 1 denoising when noisy.
                            3. Whisper **Turbo** on CUDA:0 auto-transcribes the clip unless an optional transcript override is supplied.
                            """
                        )
                        reg_name = gr.Textbox(
                            label="Voice Persona Name",
                            placeholder="e.g. cilian, narrator_1...",
                            value="My Speaker",
                        )
                        reg_audio = gr.Audio(
                            label="Speaker Audio Sample (WAV / MP3 / M4A - Auto-Trimmed to <= 10s)",
                            type="filepath",
                        )
                        reg_enhance_mode = gr.Radio(
                            choices=REG_ENHANCE_CHOICES,
                            value=REG_ENHANCE_CHOICES[0],
                            label="Reference Audio Conditioning Mode",
                        )
                        reg_transcript = gr.Textbox(
                            label="Transcript Override (Optional - leave blank for automatic Whisper Turbo ASR)",
                            placeholder="Leave empty to auto-transcribe with faster-whisper turbo...",
                            lines=2,
                        )
                        reg_btn = gr.Button("Register Voice (.breeze)", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        reg_file = gr.File(label="Exported .breeze Voice File")
                        reg_stats = gr.Markdown("Ready to register new voice persona.")

            # ---------------- TAB 4: SPEECH-TO-SPEECH CONVERSION ----------------
            with gr.TabItem("Speech-to-Speech Conversion (Mode 4)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("Transform source speech into another voice while preserving timing and cadence.")
                        with gr.Row():
                            conv_src_audio = gr.Audio(label="Source Audio (Speech to convert)", type="filepath")
                            conv_ref_audio = gr.Audio(label="Target Voice Audio (Optional if selecting below)", type="filepath")

                        with gr.Row():
                            conv_voice_dropdown = gr.Dropdown(
                                choices=initial_voices,
                                value=initial_voices[0],
                                label="Target Voice Persona (From Registry)",
                                scale=4,
                            )
                            conv_refresh_btn = gr.Button("Refresh", scale=1)

                        conv_refresh_btn.click(
                            fn=lambda: gr.update(choices=list_saved_voices()),
                            inputs=[],
                            outputs=[conv_voice_dropdown],
                        )

                        conv_ref_text = gr.Textbox(
                            label="Target Voice Transcript (Optional)",
                            placeholder="Transcript of target reference voice...",
                            lines=1,
                        )
                        with gr.Row():
                            conv_keep = gr.Slider(0, 15, value=0, step=1, label="Keep Acoustic Codebooks (0 = Full Transfer)")
                            conv_cfg = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="CFG Scale")
                            conv_seed = gr.Number(value=42, label="Random Seed", precision=0)

                        conv_btn = gr.Button("Convert Speech", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        conv_output_audio = gr.Audio(label="Converted Audio Output", type="filepath")
                        conv_stats = gr.Markdown("Ready for voice conversion.")

                conv_btn.click(
                    fn=voice_convert_generate,
                    inputs=[
                        conv_src_audio, conv_voice_dropdown, conv_ref_audio, conv_ref_text,
                        conv_keep, conv_cfg, conv_seed,
                    ],
                    outputs=[conv_output_audio, conv_stats],
                )

                reg_btn.click(
                    fn=register_voice_file,
                    inputs=[reg_name, reg_audio, reg_enhance_mode, reg_transcript],
                    outputs=[reg_file, reg_stats, tts_voice_dropdown, clone_voice_dropdown, lf_voice_dropdown, conv_voice_dropdown],
                )

            # ---------------- TAB 5: ASR & SUBTITLES TOOL ----------------
            with gr.TabItem("ASR Transcription & Subtitles (.srt)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown(
                            f"""
                            ### Whisper ASR & Subtitle Generator ({APP_CONFIG['asr_model']} on CUDA:{APP_CONFIG['asr_device']})
                            Transcribes speech of any duration using Silero VAD filtering and exports word-level SRT timestamps.
                            """
                        )
                        sub_audio = gr.Audio(label="Upload Audio File (Speech)", type="filepath")
                        sub_btn = gr.Button("Generate Subtitles & Transcript", variant="primary")

                    with gr.Column(scale=3):
                        sub_text = gr.Textbox(label="Plain Transcript", lines=5)
                        sub_srt = gr.Textbox(label="SRT Subtitles", lines=8)

                sub_btn.click(
                    fn=transcribe_subtitles_tool,
                    inputs=[sub_audio],
                    outputs=[sub_text, sub_srt],
                )

            # ---------------- TAB 6: TELEMETRY & ENGINE PROFILE ----------------
            with gr.TabItem("Engine Configuration"):
                resolved_prec = DualInstanceCluster.resolve_precision(
                    APP_CONFIG["model_path"], APP_CONFIG.get("precision")
                )
                gr.Markdown(
                    f"""
                    ### Active Engine Profile
                    - **Model GGUF**: `{APP_CONFIG['model_path']}` (**{resolved_prec.upper()}**)
                    - **Shared C++ Library**: `{APP_CONFIG['lib_path'] or 'Auto-detected'}`
                    - **Vocoder Execution**: `{APP_CONFIG['vocoder_mode'].upper()} (Step 480 / Tile 560 Isolated Per-Chunk Vocoder + RMS Leveling)`
                    - **Modular Q4 Surge**: `{'Enabled (Threshold: ' + str(APP_CONFIG['q4_threshold']) + ')' if APP_CONFIG['enable_q4_burst'] else 'Disabled (Full Precision)'}`
                    - **Default Option A Sampling**: `cfg=1.6, temp=0.55, top_p=0.85, top_k=30, rep_pen=1.08, depth_temp=0.0, dcfg=3`
                    - **Voices Directory**: `{os.path.abspath(APP_CONFIG['voices_dir'])}`
                    - **ASR Microservice**: `{'Enabled (' + APP_CONFIG['asr_model'] + ' on CUDA:' + str(APP_CONFIG['asr_device']) + ')' if APP_CONFIG['enable_asr'] else 'Disabled'}`
                    - **Enhancer Microservice**: `{'Enabled (on-demand CPU->CUDA:' + str(APP_CONFIG['enhancer_device']) + ', Auto-SNR gate @ 30 dB)' if APP_CONFIG['enable_enhancer'] else 'Disabled'}`
                    """
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Breeze-TTS Studio Decoupled Web Application")
    parser.add_argument("--model", type=str, default="breeze-tts-2-q8_0.gguf", help="Path to base GGUF model")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so")
    parser.add_argument("--precision", type=str, choices=["auto", "int8", "fp16"], default="auto", help="Engine precision mode (default: auto-detect from model path)")
    parser.add_argument("--q4-model", type=str, default=None, help="Path to modular Q4 depth decoder GGUF")
    parser.add_argument("--enable-q4", action="store_true", help="Enable modular Q4 surge tiering")
    parser.add_argument("--q4-threshold", type=int, default=10, help="Active session threshold for Q4 burst")
    parser.add_argument("--vocoder-mode", type=str, choices=["on", "dynamic", "off"], default="on", help="Vocoder mode")
    parser.add_argument("--disable-enhancer", action="store_true", help="Disable Resemble Enhancer microservice")
    parser.add_argument("--disable-asr", action="store_true", help="Disable Whisper ASR microservice")
    parser.add_argument("--enhancer-chunk-sec", type=float, default=10.0, help="Sliding window chunk duration for Resemble Enhancer")
    parser.add_argument("--enhancer-device", type=int, default=1, help="CUDA device index for Resemble Enhancer")
    parser.add_argument("--voices-dir", type=str, default="data/voices", help="Local directory for .breeze voice containers")
    parser.add_argument("--asr-device", type=int, default=0, help="CUDA device index for Whisper ASR")
    parser.add_argument("--asr-model", type=str, default="turbo", help="Whisper model size (default: turbo)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Web port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    resolved_prec = DualInstanceCluster.resolve_precision(args.model, args.precision)

    APP_CONFIG["model_path"] = args.model
    APP_CONFIG["lib_path"] = args.lib
    APP_CONFIG["precision"] = resolved_prec
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

    print("Pre-warming cluster engine, orchestrator & micro-services...")
    try:
        get_orchestrator()
        print(f"   EngineOrchestrator online (Precision: {APP_CONFIG['precision'].upper()} | Enhancer: {APP_CONFIG['enable_enhancer']} | ASR: {APP_CONFIG['enable_asr']}).")
    except Exception as e:
        print(f"   Engine initialization deferred/failed: {e}")

    demo = build_app()
    print(f"\nLaunching Breeze-TTS Studio on http://{args.host}:{args.port}")
    print(f"   Model: {APP_CONFIG['model_path']} ({APP_CONFIG['precision'].upper()})")
    print(f"   Voices Registry: {os.path.abspath(APP_CONFIG['voices_dir'])}")
    print(f"   ASR: {APP_CONFIG['asr_model']} on CUDA:{APP_CONFIG['asr_device']} (Enabled: {APP_CONFIG['enable_asr']})")
    print(f"   Enhancer: CUDA:{APP_CONFIG['enhancer_device']} (Enabled: {APP_CONFIG['enable_enhancer']})")

    allowed_dirs = [
        os.path.abspath(APP_CONFIG["voices_dir"]),
        os.path.abspath(APP_CONFIG["outputs_dir"]),
        tempfile.gettempdir(),
        os.getcwd(),
    ]
    if os.path.exists("/kaggle/working"):
        allowed_dirs.extend([
            os.path.abspath("/kaggle/working"),
            os.path.abspath("/kaggle/working/data"),
            os.path.abspath("/kaggle/working/data/outputs"),
        ])
    allowed_dirs = list({d for d in allowed_dirs if os.path.exists(d)})

    launch_kwargs: Dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "allowed_paths": allowed_dirs,
    }
    if _is_gradio_6_plus():
        launch_kwargs["css"] = ".gradio-container { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }"
        launch_kwargs["theme"] = gr.themes.Soft()

    demo.queue(default_concurrency_limit=10)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
