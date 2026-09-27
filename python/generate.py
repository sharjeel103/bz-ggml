#!/usr/bin/env python3
"""
Simple CLI generation script using bz-ggml Dual-Instance Engine.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bz_ggml import DualInstanceCluster, UserTask

def main():
    parser = argparse.ArgumentParser(description="bz-ggml Text-to-Speech Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--instruction", type=str, default="Speak clearly and naturally.", help="Instruction prompt")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so")
    parser.add_argument("--cfg-scale", type=float, default=None, help="Classifier-Free Guidance scale")
    parser.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio file for voice cloning")
    parser.add_argument("--ref-text", type=str, default=None, help="Reference transcript for voice cloning")
    parser.add_argument("--studio", action="store_true", help="Enable Studio Master mode (Full Single-Pass Vocoder, CFG 2.0, Temp 0.25)")
    parser.add_argument("--stream-pcm", action=argparse.BooleanOptionalAction, default=None, help="Stream PCM in 32-frame chunks (default: True unless --studio)")
    args = parser.parse_args()

    # Determine Studio vs Streaming settings
    is_studio = args.studio
    stream_pcm = not is_studio if args.stream_pcm is None else args.stream_pcm
    cfg_scale = (2.0 if is_studio else 1.0) if args.cfg_scale is None else args.cfg_scale

    cluster = DualInstanceCluster(args.model, args.lib, enable_q4_burst=False)
    task = UserTask(
        id=1,
        text=args.prompt,
        instruction=args.instruction,
        cfg_scale=cfg_scale,
        seed=args.seed,
        ref_text=args.ref_text,
        stream_pcm=stream_pcm
    )

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    results = cluster.run_workload([task], out_dir=out_dir)
    cluster.close()

    if results:
        res = results[0]
        if res.wav_path and os.path.exists(res.wav_path) and res.wav_path != args.output:
            os.rename(res.wav_path, args.output)
        mode_str = "Studio Master (Full Single-Pass)" if not stream_pcm else "Streaming (32-frame chunks)"
        print(f"\n[{mode_str}] Synthesized {res.audio_s:.2f}s of audio in {res.compute_wall_s:.2f}s (RTF={res.rtf:.3f}, TTFA={res.ttfa_s:.3f}s)")
        print(f"Saved output to: {args.output}")

if __name__ == "__main__":
    main()
