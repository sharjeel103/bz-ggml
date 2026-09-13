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
    args = parser.parse_args()

    cluster = DualInstanceCluster(args.model, args.lib)
    task = UserTask(id=1, text=args.prompt, instruction=args.instruction)

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    results = cluster.run_workload([task], out_dir=out_dir)
    cluster.close()

    if results:
        res = results[0]
        if res.wav_path != args.output:
            os.rename(res.wav_path, args.output)
        print(f"\nSynthesized {res.audio_s:.2f}s of audio in {res.compute_wall_s:.2f}s (RTF={res.rtf:.3f}, TTFA={res.ttfa_s:.3f}s)")
        print(f"Saved output to: {args.output}")

if __name__ == "__main__":
    main()
