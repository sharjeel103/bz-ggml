#!/usr/bin/env python3
"""
Breeze-TTS Studio: Pure FP16 Gradio Web Application
Provides in-browser Text-to-Speech, Zero-Shot Voice Cloning, Voice Design,
and Speech-to-Speech Voice Conversion with Studio Single-Pass vs Streaming controls.
"""

import argparse
import os
import sys
import tempfile
import time
from typing import Optional, Tuple, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bz_ggml import (
    DualInstanceCluster, UserTask, VoiceConversionTask,
    load_breeze_voice, save_breeze_voice
)
from bz_ggml.asr import ASRService

import gradio as gr


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


# Global cluster and ASR instances
cluster_instance: Optional[DualInstanceCluster] = None
current_model_path: Optional[str] = None
current_lib_path: Optional[str] = None
asr_service_instance: Optional[ASRService] = None


def get_asr_service(cuda_device: int = 1) -> ASRService:
    """Lazy-initializes ASR service pinned strictly to GPU 1."""
    global asr_service_instance
    if asr_service_instance is None:
        asr_service_instance = ASRService(model_size="small", cuda_device=cuda_device, compute_type="int8_float16")
    return asr_service_instance


def transcribe_audio_sample(audio_path: Optional[str]) -> str:
    """Transcribes an audio sample on GPU 1 using faster-whisper-small."""
    if not audio_path or not os.path.exists(audio_path):
        return "Error: Please upload or record an audio file first."
    try:
        asr = get_asr_service(cuda_device=1)
        transcript = asr.transcribe(audio_path)
        return transcript.strip()
    except Exception as e:
        return f"ASR Transcription Error: {str(e)}"


def register_voice_file(
    voice_name: str,
    audio_path: Optional[str],
    transcript: str,
    model_path: str
) -> Tuple[Optional[str], str]:
    """Encodes voice reference on GPU 0 and saves into standard .breeze container."""
    if not voice_name.strip():
        return None, "❌ Error: Voice name is required."
    if not audio_path or not os.path.exists(audio_path):
        return None, "❌ Error: Audio sample file is required."
    if not transcript.strip():
        return None, "❌ Error: Exact verbatim transcript is required for voice registration."

    try:
        cluster = get_cluster(model_path, current_lib_path)
        print(f"[Register Voice] Loading audio: {audio_path}")
        pcm = load_audio_24k(audio_path)
        t0 = time.time()
        codes, frames = cluster.encode_audio(pcm, gpu_id=0)
        enc_time = time.time() - t0

        out_dir = tempfile.mkdtemp(prefix="breeze_voice_")
        safe_name = "".join(c for c in voice_name.strip() if c.isalnum() or c in (' ', '_', '-')).rstrip()
        breeze_filename = f"{safe_name.lower().replace(' ', '_')}.breeze"
        breeze_path = os.path.join(out_dir, breeze_filename)

        save_breeze_voice(
            path=breeze_path,
            text=transcript.strip(),
            codes=codes,
            frames=frames,
            sample_rate=24000,
            n_codebooks=16
        )

        file_kb = os.path.getsize(breeze_path) / 1024.0
        msg = (
            f"✅ **Voice Registered Successfully!**\n\n"
            f"• **Voice Name**: `{voice_name.strip()}`\n"
            f"• **Frames**: {frames} ({frames * 0.08:.2f}s acoustic reference)\n"
            f"• **Tokens**: {len(codes)} (16 codebooks @ 12.5 Hz)\n"
            f"• **Encode Latency**: {enc_time:.2f}s (Pinned to GPU 0 Audio Encoder Pool)\n"
            f"• **Container Size**: {file_kb:.1f} KB (`.breeze` format)\n"
            f"• **Ready for Zero-Shot Cloning**: Instant zero-latency loading with 0 MB audio re-encoding!"
        )
        return breeze_path, msg
    except Exception as e:
        return None, f"❌ Voice Registration Failed: {str(e)}"


def get_cluster(model_path: str, lib_path: Optional[str] = None) -> DualInstanceCluster:
    global cluster_instance, current_model_path, current_lib_path
    if cluster_instance is None or current_model_path != model_path:
        if cluster_instance is not None:
            cluster_instance.close()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"[Gradio Studio] Initializing DualInstanceCluster in Pure FP16 mode: {model_path}")
        cluster_instance = DualInstanceCluster(
            model_path=model_path,
            lib_path=lib_path,
            enable_q4_burst=False,  # Enforce 100% Pure FP16 precision
            enable_vocoder=True,
            vocoder_mode="on"
        )
        current_model_path = model_path
        current_lib_path = lib_path
    return cluster_instance


def tts_generate(
    text: str,
    instruction: str,
    delivery_mode: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    max_steps: int,
    model_path: str
) -> Tuple[Optional[str], str]:
    if not text.strip():
        return None, "Error: Text input is empty."

    stream_pcm = (delivery_mode != "Studio Master (Full Single-Pass, Zero Clicks)")
    out_dir = tempfile.mkdtemp(prefix="breeze_tts_")

    try:
        cluster = get_cluster(model_path, current_lib_path)
        task = UserTask(
            id=int(time.time() * 1000) % 100000,
            text=text,
            instruction=instruction,
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
        mode_label = "Studio Master (Full Single-Pass FP16)" if not stream_pcm else "Real-Time Streaming (32-frame chunks)"
        stats_msg = (
            f"✅ **Success** | {mode_label}\n\n"
            f"• **Audio Length**: {res.audio_s:.2f}s ({int(res.audio_s * 24000)} samples)\n"
            f"• **Compute Time**: {res.compute_wall_s:.2f}s (Total: {wall:.2f}s)\n"
            f"• **Real-Time Factor (RTF)**: {res.rtf:.3f}x\n"
            f"• **Worker Island**: {res.worker}\n"
            f"• **Precision**: Pure FP16 (FP32 Accumulators)"
        )
        return res.wav_path, stats_msg
    except Exception as e:
        return None, f"❌ Error during synthesis: {str(e)}"


def voice_clone_generate(
    text: str,
    ref_breeze_file: Optional[str],
    ref_audio: Optional[str],
    ref_text: str,
    instruction: str,
    delivery_mode: str,
    cfg_scale: float,
    temperature: float,
    seed: int,
    max_steps: int,
    model_path: str
) -> Tuple[Optional[str], str]:
    if not text.strip():
        return None, "Error: Target text is empty."

    # Priority 1: Pre-registered .breeze container file (instant load, 0 MB re-encoding)
    if ref_breeze_file and os.path.exists(ref_breeze_file):
        try:
            v_data = load_breeze_voice(ref_breeze_file)
            ref_codes = v_data["codes"]
            ref_frames = v_data["frames"]
            final_ref_text = v_data["text"]
            source_desc = f"Pre-registered Voice Container (`{os.path.basename(ref_breeze_file)}`)"
        except Exception as e:
            return None, f"❌ Error loading .breeze file: {str(e)}"
    elif ref_audio and os.path.exists(ref_audio):
        if not ref_text.strip():
            return None, "❌ Error: Exact verbatim reference transcript is required when using raw audio."
        try:
            cluster = get_cluster(model_path, current_lib_path)
            ref_pcm = load_audio_24k(ref_audio)
            ref_codes, ref_frames = cluster.encode_audio(ref_pcm, gpu_id=0)
            final_ref_text = ref_text.strip()
            source_desc = f"Raw Audio Sample (`{os.path.basename(ref_audio)}`)"
        except Exception as e:
            return None, f"❌ Error encoding reference audio: {str(e)}"
    else:
        return None, "❌ Error: Please provide either a registered .breeze voice container or an audio sample with transcript."

    stream_pcm = (delivery_mode != "Studio Master (Full Single-Pass, Zero Clicks)")
    out_dir = tempfile.mkdtemp(prefix="breeze_clone_")

    try:
        cluster = get_cluster(model_path, current_lib_path)
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
        mode_label = "Studio Master (Full Single-Pass FP16)" if not stream_pcm else "Real-Time Streaming (32-frame chunks)"
        stats_msg = (
            f"✅ **Cloned Successfully** | {mode_label}\n\n"
            f"• **Reference Voice**: {source_desc}\n"
            f"• **Audio Length**: {res.audio_s:.2f}s\n"
            f"• **Compute Time**: {res.compute_wall_s:.2f}s (Total: {wall:.2f}s)\n"
            f"• **Real-Time Factor (RTF)**: {res.rtf:.3f}x\n"
            f"• **Precision**: Pure FP16 (No Q4 Quantization)"
        )
        return res.wav_path, stats_msg
    except Exception as e:
        return None, f"❌ Error during voice cloning: {str(e)}"


def voice_convert_generate(
    source_audio: Optional[str],
    ref_audio: Optional[str],
    ref_text: str,
    keep_acoustic: int,
    cfg_scale: float,
    seed: int,
    model_path: str
) -> Tuple[Optional[str], str]:
    if not source_audio:
        return None, "Error: Source audio file is required."
    if not ref_audio:
        return None, "Error: Target reference voice audio is required."

    out_dir = tempfile.mkdtemp(prefix="breeze_conv_")

    try:
        cluster = get_cluster(model_path, current_lib_path)
        # Mode 4 runs via convert_voice_task
        task = VoiceConversionTask(
            id=int(time.time() * 1000) % 100000,
            src_codes=[],
            src_frames=0,
            ref_text=ref_text.strip() if ref_text else None,
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
        import numpy as np

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
            f"• **Worker**: {stats['worker']}\n"
            f"• **Precision**: Pure FP16 Single-Pass"
        )
        return out_wav, stats_msg
    except Exception as e:
        return None, f"❌ Error during voice conversion: {str(e)}"


def build_app(default_model: str) -> gr.Blocks:
    custom_css = """
    .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .status-badge { font-weight: bold; color: #10b981; }
    """

    with gr.Blocks(title="Breeze-TTS Studio (Pure FP16)", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ Breeze-TTS Studio & Voice Lab
            ### High-Fidelity Neural Speech Synthesis & Voice Cloning in Pure FP16
            *Zero Quantization Degradation | Single-Pass Studio Vocoder | 24kHz Broadcast Quality*
            """
        )

        with gr.Row():
            model_input = gr.Textbox(
                label="GGUF Model Path",
                value=default_model,
                placeholder="Path to breeze-base.gguf",
                scale=4
            )
            with gr.Column(scale=1):
                precision_badge = gr.Markdown(
                    "**Active Precision:** `Pure FP16 Weights (FP32 Accumulators)`\n\n"
                    "**Modular Q4 Surge:** `Disabled (100% Fidelity)`"
                )

        with gr.Tabs():
            # ---------------- TAB 1: TEXT-TO-SPEECH & VOICE DESIGN ----------------
            with gr.TabItem("✨ Text-to-Speech & Voice Design"):
                with gr.Row():
                    with gr.Column(scale=3):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Type or paste the script you want synthesized for your YouTube video...",
                            lines=4,
                            value="Welcome back to the channel. Today we are diving into the most exciting breakthroughs in artificial intelligence."
                        )
                        tts_instruction = gr.Textbox(
                            label="Speaker Persona / Emotional Instruction",
                            placeholder="Describe tone, accent, pacing, or emotional style...",
                            value="Speak with a warm, authoritative, engaging documentary narration style."
                        )

                        with gr.Row():
                            tts_delivery = gr.Radio(
                                choices=[
                                    "Studio Master (Full Single-Pass, Zero Clicks)",
                                    "Real-Time Streaming (32-Frame Chunks)"
                                ],
                                value="Studio Master (Full Single-Pass, Zero Clicks)",
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
                        tts_text, tts_instruction, tts_delivery,
                        tts_cfg, tts_temp, tts_seed, tts_max, model_input
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
                            value="This is a test of zero-shot voice cloning with perfect speaker identity preservation."
                        )
                        
                        with gr.Group():
                            gr.Markdown("#### Option A: Use Pre-Registered Voice Container (`.breeze`) — Instant 0ms Load")
                            clone_ref_breeze = gr.File(
                                label="Upload .breeze Voice File",
                                file_types=[".breeze"],
                                type="filepath"
                            )

                        with gr.Group():
                            gr.Markdown("#### Option B: Upload Raw Audio & Auto-Transcribe (GPU 1 ASR)")
                            clone_ref_audio = gr.Audio(
                                label="Reference Voice Sample (4 to 7 seconds WAV/MP3 recommended)",
                                type="filepath"
                            )
                            with gr.Row():
                                clone_asr_btn = gr.Button("🎙️ Auto-Transcribe Audio (GPU 1 Whisper ASR)", variant="secondary")
                            clone_ref_text = gr.Textbox(
                                label="Reference Transcript (Must match verbatim. Click Auto-Transcribe or edit manually)",
                                placeholder="Speaker verbatim transcript...",
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
                                    "Studio Master (Full Single-Pass, Zero Clicks)",
                                    "Real-Time Streaming (32-Frame Chunks)"
                                ],
                                value="Studio Master (Full Single-Pass, Zero Clicks)",
                                label="Vocoder Synthesis Mode"
                            )

                        with gr.Accordion("Advanced Voice Cloning Parameters", open=False):
                            with gr.Row():
                                clone_cfg = gr.Slider(1.0, 3.0, value=1.0, step=0.1, label="CFG Guidance Scale (1.0 = Default, 1.5 = Enhanced)")
                                clone_temp = gr.Slider(0.1, 1.2, value=0.25, step=0.05, label="Temperature")
                                clone_seed = gr.Number(value=42, label="Random Seed")
                                clone_max = gr.Slider(100, 2000, value=1000, step=50, label="Max Output Frames")

                        clone_btn = gr.Button("▶ Clone Voice & Synthesize", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        clone_output_audio = gr.Audio(label="Cloned Speech Output (24 kHz)", type="filepath")
                        clone_stats = gr.Markdown("Provide a registered `.breeze` voice container or upload reference audio to begin.")

                clone_btn.click(
                    fn=voice_clone_generate,
                    inputs=[
                        clone_text, clone_ref_breeze, clone_ref_audio, clone_ref_text, clone_instruction,
                        clone_delivery, clone_cfg, clone_temp, clone_seed, clone_max, model_input
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
                            3. **Review & Correct** transcript to ensure 100% phonetic accuracy.
                            4. **Register**: Encodes reference tokens on GPU 0 and exports a portable `.breeze` container.
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
                            label="Verbatim Transcript (Human in the Loop Review)",
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
                        reg_file = gr.File(label="Exported .breeze Voice File (Download & Reuse)")
                        reg_stats = gr.Markdown("Ready to register new voice persona.")

                reg_btn.click(
                    fn=register_voice_file,
                    inputs=[reg_name, reg_audio, reg_transcript, model_input],
                    outputs=[reg_file, reg_stats]
                )

            # ---------------- TAB 4: SPEECH-TO-SPEECH CONVERSION ----------------
            with gr.TabItem("🔄 Speech-to-Speech Voice Conversion (Mode 4)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("Transform one speaker's voice into another while preserving timing, cadence, and pronunciation.")
                        with gr.Row():
                            conv_src_audio = gr.Audio(label="Source Audio (Speech to convert)", type="filepath")
                            conv_ref_audio = gr.Audio(label="Target Voice (Voice to convert into)", type="filepath")
                        conv_ref_text = gr.Textbox(
                            label="Target Voice Transcript (Optional but recommended)",
                            placeholder="Transcript of the target reference voice...",
                            lines=1
                        )
                        with gr.Row():
                            conv_keep = gr.Slider(
                                0, 15, value=0, step=1,
                                label="Keep Acoustic Codebooks (0=Full Voice Transfer, 1..3=Keep Source Pitch/Melody)"
                            )
                            conv_cfg = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="CFG Guidance Scale")
                            conv_seed = gr.Number(value=42, label="Random Seed")

                        conv_btn = gr.Button("▶ Convert Speech", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        conv_output_audio = gr.Audio(label="Converted Audio Output", type="filepath")
                        conv_stats = gr.Markdown("Ready for voice conversion.")

                conv_btn.click(
                    fn=voice_convert_generate,
                    inputs=[
                        conv_src_audio, conv_ref_audio, conv_ref_text,
                        conv_keep, conv_cfg, conv_seed, model_input
                    ],
                    outputs=[conv_output_audio, conv_stats]
                )

            # ---------------- TAB 5: TELEMETRY & HARDWARE ----------------
            with gr.TabItem("📊 Hardware & Precision Telemetry"):
                gr.Markdown(
                    """
                    ### Active Architecture & Precision Profile
                    * **Model Precision**: Pure 16-bit Float (`GGML_TYPE_F16`) across all layers.
                    * **Compute Precision**: 32-bit Floating-Point (`float32`) for attention, layer norm, and vocoder activations.
                    * **Surge INT4 Offload**: Disabled (`enable_q4_burst=False`) to guarantee zero quantization error.
                    * **Vocoder Architecture**: 8-layer Causal Pre-Transformer (72-frame sliding window) + $1920\times$ Cascaded Transposed Convolutions + SnakeBeta.
                    * **Audio Output**: 24,000 Hz, 16-bit / 32-bit linear PCM.
                    """
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Breeze-TTS Studio Web Application")
    parser.add_argument("--model", type=str, default="breeze-base.gguf", help="Path to GGUF model")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Web port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    global current_lib_path
    current_lib_path = args.lib

    demo = build_app(default_model=args.model)
    print(f"\n🚀 Launching Breeze-TTS Studio on http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
