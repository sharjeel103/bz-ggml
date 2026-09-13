#!/usr/bin/env python3
import os
import sys
import time
import argparse
from bz_ggml import DualInstanceCluster, UserTask, ClusterTelemetry

PROMPTS = [
    # 5 Short requests (~5-10 words, should finish in < 3s!)
    "Good morning! Your order has been confirmed.",
    "System update completed successfully without errors.",
    "Flight seven twelve to San Francisco is now boarding.",
    "Your payment was processed and receipt sent.",
    "Thank you very much. Have a wonderful day.",

    # 10 Medium requests (~30-50 words)
    "The project planning meeting has been rescheduled to Thursday at three in the afternoon, conference room four.",
    "The air quality index is currently moderate across the valley with light northwesterly winds throughout the morning.",
    "All cluster network services are running normally with optimal bandwidth and zero packet loss detected across all nodes.",
    "Please remember to submit your weekly engineering progress report before five this evening for team review.",
    "Your ride has arrived outside the main hotel lobby. The silver vehicle is waiting with its hazard lights blinking.",
    "Security patches have been safely installed across all microservices, and database replication is functioning normally.",
    "The temperature today will peak at seventy-eight degrees Fahrenheit under mostly clear skies with low humidity.",
    "Automated unit and integration test suites passed completely on branch main with full line coverage and zero regressions.",
    "The conference registration desk is open on the second floor adjacent to the grand ballroom for badge pickup.",
    "Scheduled database maintenance will begin tonight at midnight and is expected to conclude within forty-five minutes.",

    # 5 Long requests (~80-150 words)
    "Artificial intelligence infrastructure requires careful balance between memory bandwidth, arithmetic intensity, and tensor core utilization. When deploying speech generation models on dual Tesla T4 GPUs, coarse-grained serial queue pools inevitably cause head-of-line blocking where brief phrases wait behind essays. By introducing continuous iteration-level multi-session round-robin scheduling, every active user receives immediate acoustic tokens, cutting latency by ninety-nine percent.",
    "Speech synthesis workflows rely on four distinct neural stages. First, the text tokenizer and text encoder transform raw input characters into continuous phonetic embeddings. Next, the twenty-eight layer causal backbone transformer models macro-temporal prosody and predicts rhythm tokens. Then, the twelve-layer depth decoder predicts fifteen fine-grained acoustic codebooks per frame. Finally, the neural vocoder converts discrete tokens into high-fidelity twenty-four kilohertz audio.",
    "Classifier-free guidance provides precise stylistic steering for synthesized speech. By evaluating both a conditioned branch containing emotional instructions and an unconditioned neutral branch simultaneously, the mathematical difference is scaled to steer the final audio toward desired expressiveness. In modern speech engines, this multi-branch evaluation is batched inside tensor cores to preserve real-time throughput.",
    "In large language model and neural audio serving, the memory bandwidth bottleneck dictates single-token latency. Because each generated frame fetches gigabytes of model weights from memory to registers, 4-bit and 8-bit quantization directly halves memory traffic, allowing multiple concurrent user streams to share memory bus bandwidth simultaneously.",
    "Our distributed cluster architecture assigns dedicated generation workers to GPU 0 and GPU 1, while reserving GPU 1 for the dedicated neural vocoder. By transmitting only sixty-four bytes of discrete codebook tokens between GPUs, inter-GPU bus latency is completely eliminated, allowing uninterrupted twenty-four kilohertz PCM streaming to twenty users concurrently."
]

def main():
    parser = argparse.ArgumentParser(description="20-User Dynamic Multi-Session Burst Benchmark")
    parser.add_argument("--model", type=str, default="/kaggle/input/breeze-tts-2-q8-0/breeze-tts-2-q8_0.gguf", help="Path to GGUF model")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so")
    parser.add_argument("--out-dir", type=str, default="elastic_20user_audio", help="Output audio directory")
    parser.add_argument("--n-users", type=int, default=20, help="Number of concurrent users")
    parser.add_argument("--quantum", type=int, default=2, help="Adaptive quantum frame count (default: 2)")
    parser.add_argument("--enable-q4-burst", action="store_true", help="Enable surge Q4 tiering for high load")
    parser.add_argument("--q4-model", type=str, default=None, help="Path to Q4 GGUF model for surge tiering")
    parser.add_argument("--q4-threshold", type=int, default=10, help="Per-GPU slot threshold to trigger Q4 tiering")
    args = parser.parse_args()

    print("================================================================================")
    print(f"      DYNAMIC MULTI-SESSION 10/10 CLUSTER: {args.n_users}-USER BURST TEST")
    print("================================================================================")
    print(f"Primary Model:   {args.model}")
    if args.enable_q4_burst:
        print(f"Surge Q4 Model:  {args.q4_model} (Trigger Threshold: {args.q4_threshold} slots/GPU)")
    else:
        print("Surge Q4 Tier:   Disabled (100% Q8_0 High-Fidelity)")
    print(f"Quantum Frames:  {args.quantum} (Initial TTFA: 1 frame)")
    print(f"Output:          {args.out_dir}")

    cluster = DualInstanceCluster(
        model_path=args.model,
        lib_path=args.lib,
        q4_model_path=args.q4_model,
        enable_q4_burst=args.enable_q4_burst,
        q4_threshold=args.q4_threshold
    )
    telemetry = ClusterTelemetry(interval_s=0.2)
    telemetry.start()

    tasks = []
    for i in range(args.n_users):
        text = PROMPTS[i % len(PROMPTS)]
        tasks.append(UserTask(id=i+1, text=text, instruction="Speak naturally and expressively.", seed=42 + i))

    print(f"\n[Burst] Launching {len(tasks)} concurrent user tasks simultaneously at t = 0.00s...")
    t_start = time.time()

    def progress_cb(r):
        print(f"  [EOS Finished] User {r.id:02d} ({r.words:3d}w) | Audio: {r.audio_s:5.2f}s | Wall: {r.compute_wall_s:5.2f}s | TTFA: {r.ttfa_s:5.3f}s | RTF: {r.rtf:5.3f} | {r.worker}")

    results = cluster.run_workload(
        tasks,
        out_dir=args.out_dir,
        on_progress=progress_cb,
        quantum_frames=args.quantum
    )
    total_wall = time.time() - t_start

    telemetry.stop()
    telem_stats = telemetry.get_summary()

    total_audio = sum(r.audio_s for r in results)
    avg_ttfa = sum(r.ttfa_s for r in results) / len(results)
    max_ttfa = max(r.ttfa_s for r in results)
    min_ttfa = min(r.ttfa_s for r in results)
    cluster_rtf = total_wall / total_audio if total_audio > 0 else 0.0
    realtime_mult = total_audio / total_wall if total_wall > 0 else 0.0

    print("\n================================================================================")
    print("                      CLUSTER PERFORMANCE REPORT")
    print("================================================================================")
    print(f"Total Requests:           {len(results)} users")
    print(f"Cluster Wall Time:        {total_wall:.2f} s ({total_wall/60.0:.2f} min)")
    print(f"Total Speech Synthesized: {total_audio:.2f} s ({total_audio/60.0:.2f} min)")
    print(f"Cluster Real-Time Speed:  {realtime_mult:.2f}x Real-Time (RTF: {cluster_rtf:.3f})")
    print(f"Mean TTFA:                {avg_ttfa:.3f} s")
    print(f"Min TTFA:                 {min_ttfa:.3f} s")
    print(f"Max TTFA:                 {max_ttfa:.3f} s")
    print("--------------------------------------------------------------------------------")
    print("GPU Hardware Telemetry:")
    for dev_id, s in telem_stats.items():
        print(f"  GPU {dev_id}: Util Mean = {s.get('util_mean_pct', 0.0):.1f}% | Util Max = {s.get('util_max_pct', 0.0):.1f}% | VRAM Peak = {s.get('mem_max_mib', 0.0):.1f} MiB")
    print("================================================================================")

    cluster.close()

if __name__ == "__main__":
    main()
