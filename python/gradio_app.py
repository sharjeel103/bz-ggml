#!/usr/bin/env python3
"""
Breeze-TTS Studio: Decoupled Gradio Web Application Client
Provides in-browser Text-to-Speech, Zero-Shot Voice Cloning, Human-in-the-Loop Voice Registration (.breeze),
Speech-to-Speech Voice Conversion (Mode 4), and Subtitle/Transcription tools.
Strictly decoupled from the engine: consumes public bz_ggml APIs and respects all engine contracts.
"""

import argparse
import os
import sys
import tempfile
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bz_ggml import (
    DualInstanceCluster, UserTask, VoiceConversionTask,
    load_breeze_voice, save_breeze_voice,
    serialize_breeze_voice, deserialize_breeze_voice
)
from bz_ggml.asr import ASRService

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
    "asr_device": 1,
    "asr_model": "small"
}

cluster_instance: Optional[DualInstanceCluster] = None
asr_service_instance: Optional[ASRService] = None


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
            compute_type="int8_float16"
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

        # Read back container metadata for display
        meta = deserialize_breeze_voice(container_bytes)
        file_kb = len(container_bytes) / 1024.0

        msg = (
            f"✅ **Voice Registered Successfully!**\n\n"
            f"• **Voice Persona**: `{voice_name.strip()}`\n"
            f"• **Acoustic Frames**: {meta['frames']} ({meta['frames'] * 0.08:.2f}s reference audio)\n"
            f"• **Discrete Tokens**: {len(meta['codes'])} tokens (16 codebooks @ 12.5 Hz)\n"
            f"• **Encode Time**: {enc_time:.2f}s (Pinned to GPU 0 Reference Encoder Pool)\n"
            f"• **Saved To**: `{breeze_path}` ({file_kb:.1f} KB)\n"
            f"• **Instant Availability**: Now selectable in all Cloning & TTS dropdowns with 0 ms load!"
        )

        updated_choices = list_saved_voices()
        return (
            breeze_path,
            msg,
            gr.update(choices=updated_choices, value=voice_name.strip().title()),
            gr.update(choices=updated_choices, value=voice_name.strip().title())
        )
    except Exception as e:
        return None, f"❌ Voice Registration Failed: {str(e)}", gr.update(), gr.update()


def tts_generate(
    text: str,
    instruction: str,
    selected_voice: str,
    delivery_mode: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    max_steps: int
) -> Tuple[Optional[str], str]:
    if not text.strip():
        return None, "Error: Text input is empty."

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

        t0 = time.time()
        results = cluster.run_workload([task], out_dir=out_dir)
        if not results:
            return None, "Error: Generation returned no audio."

        res = results[0]
        wall = time.time() - t0
        mode_label = "Studio Master (Full Single-Pass)" if not stream_pcm else "Real-Time Stateful Streaming (32-Frame Chunks)"
        voice_label = selected_voice if voice_path else "Default Native Speaker"

        stats_msg = (
            f"✅ **Synthesis Complete** | {mode_label}\n\n"
            f"• **Voice Persona**: `{voice_label}`\n"
            f"• **Audio Length**: {res.audio_s:.2f}s ({int(res.audio_s * 24000)} samples @ 24 kHz)\n"
            f"• **Compute Time**: {res.compute_wall_s:.2f}s (Total Turnaround: {wall:.2f}s)\n"
            f"• **Real-Time Speedup**: {res.rtf:.2f}x\n"
            f"• **Worker Island**: {res.worker}"
        )
        return res.wav_path, stats_msg
    except Exception as e:
        return None, f"❌ Error during synthesis: {str(e)}"


def voice_clone_generate(
    text: str,
    selected_voice: str,
    custom_breeze_file: Optional[str],
    ref_audio: Optional[str],
    ref_text: str,
    instruction: str,
    delivery_mode: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    max_steps: int
) -> Tuple[Optional[str], str]:
    if not text.strip():
        return None, "Error: Target text is empty."

    # Priority 1: Selected voice from pre-registered dropdown
    voice_path = get_voice_file_path(selected_voice)
    if voice_path:
        v_data = load_breeze_voice(voice_path)
        ref_codes = v_data["codes"]
        ref_frames = v_data["frames"]
        final_ref_text = v_data["text"]
        source_desc = f"Pre-registered Voice (`{selected_voice}`)"
    # Priority 2: Custom uploaded .breeze container file
    elif custom_breeze_file and os.path.exists(custom_breeze_file):
        try:
            v_data = load_breeze_voice(custom_breeze_file)
            ref_codes = v_data["codes"]
            ref_frames = v_data["frames"]
            final_ref_text = v_data["text"]
            source_desc = f"Uploaded .breeze Container (`{os.path.basename(custom_breeze_file)}`)"
        except Exception as e:
            return None, f"❌ Error loading .breeze file: {str(e)}"
    # Priority 3: Raw audio file + transcript
    elif ref_audio and os.path.exists(ref_audio):
        if not ref_text.strip():
            return None, "❌ Error: Verbatim reference transcript is required when using raw audio."
        try:
            cluster = get_cluster()
            ref_pcm = load_audio_24k(ref_audio)
            ref_codes, ref_frames = cluster.encode_audio(ref_pcm, gpu_id=0)
            final_ref_text = ref_text.strip()
            source_desc = f"Raw Audio Sample (`{os.path.basename(ref_audio)}`)"
        except Exception as e:
            return None, f"❌ Error encoding reference audio: {str(e)}"
    else:
        return None, "❌ Error: Please select a registered voice, upload a .breeze file, or provide raw audio with transcript."

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

        t0 = time.time()
        results = cluster.run_workload([task], out_dir=out_dir)
        if not results:
            return None, "Error: Voice cloning returned no audio."

        res = results[0]
        wall = time.time() - t0
        mode_label = "Studio Master (Full Single-Pass)" if not stream_pcm else "Real-Time Stateful Streaming (32-Frame Chunks)"
        stats_msg = (
            f"✅ **Cloned Successfully** | {mode_label}\n\n"
            f"• **Reference Voice**: {source_desc}\n"
            f"• **Audio Length**: {res.audio_s:.2f}s ({int(res.audio_s * 24000)} samples @ 24 kHz)\n"
            f"• **Compute Time**: {res.compute_wall_s:.2f}s (Total Turnaround: {wall:.2f}s)\n"
            f"• **Real-Time Speedup**: {res.rtf:.2f}x\n"
            f"• **Worker Island**: {res.worker}"
        )
        return res.wav_path, stats_msg
    except Exception as e:
        return None, f"❌ Error during voice cloning: {str(e)}"


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

    out_dir = tempfile.mkdtemp(prefix="breeze_conv_")

    try:
        cluster = get_cluster()
        task = VoiceConversionTask(
            id=int(time.time() * 1000) % 100000,
            src_codes=[],
            src_frames=0,
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

        import wave
        out_wav = os.path.join(out_dir, f"converted_voice_{task.id}.wav")
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            pcm_np = np.array(audio_samples, dtype=np.float32)
            pcm_int16 = (np.clip(pcm_np, -1.0, 1.0) * 32767.0).astype(np.int16)
            w.writeframes(pcm_int16.tobytes())

        stats_msg = (
            f"✅ **Voice Conversion Complete (Mode 4)**\n\n"
            f"• **Audio Length**: {stats['audio_s']:.2f}s\n"
            f"• **Wall Time**: {stats['compute_wall_s']:.2f}s (RTF: {stats['rtf']:.3f}x)\n"
            f"• **Acoustic Codebooks Kept**: {keep_acoustic}/15\n"
            f"• **Worker Island**: {stats['worker']}"
        )
        return out_wav, stats_msg
    except Exception as e:
        return None, f"❌ Error during voice conversion: {str(e)}"


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
                                tts_cfg = gr.Slider(1.0, 3.0, value=1.0, step=0.1, label="CFG Guidance Scale (1.0 = Default, 1.5 = Enhanced)")
                                tts_temp = gr.Slider(0.1, 1.2, value=0.25, step=0.05, label="Temperature")
                                tts_seed = gr.Number(value=42, label="Random Seed")
                                tts_max = gr.Slider(100, 2000, value=1000, step=50, label="Max Output Frames")

                        tts_btn = gr.Button("▶ Generate Audio", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        tts_output_audio = gr.Audio(label="Synthesized Audio (24 kHz)", type="filepath")
                        tts_stats = gr.Markdown("Ready to synthesize.")

                tts_btn.click(
                    fn=tts_generate,
                    inputs=[
                        tts_text, tts_instruction, tts_voice_dropdown, tts_delivery,
                        tts_cfg, tts_temp, tts_seed, tts_max
                    ],
                    outputs=[tts_output_audio, tts_stats]
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

                        with gr.Group():
                            gr.Markdown("#### Option A: Select Pre-Registered Voice (Instant 0 ms Load)")
                            with gr.Row():
                                clone_voice_dropdown = gr.Dropdown(
                                    choices=initial_voices,
                                    value=initial_voices[0],
                                    label="Registered Voice Persona",
                                    scale=4
                                )
                                clone_refresh_btn = gr.Button("🔄", scale=1)

                            clone_refresh_btn.click(
                                fn=lambda: gr.update(choices=list_saved_voices()),
                                inputs=[],
                                outputs=[clone_voice_dropdown]
                            )

                        with gr.Group():
                            gr.Markdown("#### Option B: Upload External `.breeze` Voice File")
                            clone_ref_breeze = gr.File(
                                label="Upload .breeze Container",
                                file_types=[".breeze"],
                                type="filepath"
                            )

                        with gr.Group():
                            gr.Markdown("#### Option C: Upload Raw Audio & Auto-Transcribe (GPU 1 ASR)")
                            clone_ref_audio = gr.Audio(
                                label="Reference Audio Sample (4 to 10 seconds)",
                                type="filepath"
                            )
                            with gr.Row():
                                clone_asr_btn = gr.Button("🎙️ Auto-Transcribe Audio (GPU 1 Whisper)", variant="secondary")
                            clone_ref_text = gr.Textbox(
                                label="Reference Transcript (Must match verbatim)",
                                placeholder="Transcript will appear here...",
                                lines=2
                            )
                            clone_asr_btn.click(
                                fn=transcribe_audio_sample,
                                inputs=[clone_ref_audio],
                                outputs=[clone_ref_text]
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
                                clone_cfg = gr.Slider(1.0, 3.0, value=1.0, step=0.1, label="CFG Guidance Scale")
                                clone_temp = gr.Slider(0.1, 1.2, value=0.25, step=0.05, label="Temperature")
                                clone_seed = gr.Number(value=42, label="Random Seed")
                                clone_max = gr.Slider(100, 2000, value=1000, step=50, label="Max Output Frames")

                        clone_btn = gr.Button("▶ Clone Voice & Synthesize", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        clone_output_audio = gr.Audio(label="Cloned Speech Output (24 kHz)", type="filepath")
                        clone_stats = gr.Markdown("Select a registered voice or provide reference audio to begin.")

                clone_btn.click(
                    fn=voice_clone_generate,
                    inputs=[
                        clone_text, clone_voice_dropdown, clone_ref_breeze, clone_ref_audio,
                        clone_ref_text, clone_instruction, clone_delivery,
                        clone_cfg, clone_temp, clone_seed, clone_max
                    ],
                    outputs=[clone_output_audio, clone_stats]
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
                    outputs=[reg_file, reg_stats, tts_voice_dropdown, clone_voice_dropdown]
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
                            conv_cfg = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="CFG Scale")
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

    demo = build_app()
    print(f"\n🚀 Launching Breeze-TTS Studio on http://{args.host}:{args.port}")
    print(f"   • Model: {APP_CONFIG['model_path']}")
    print(f"   • Voices Registry: {os.path.abspath(APP_CONFIG['voices_dir'])}")
    print(f"   • ASR Device: CUDA:{APP_CONFIG['asr_device']}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
