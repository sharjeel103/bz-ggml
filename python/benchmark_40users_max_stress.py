#!/usr/bin/env python3
import os
import sys
import time
import argparse
import random
from bz_ggml import DualInstanceCluster, UserTask, ClusterTelemetry

# 40 Diverse technical, scientific, historical, and philosophical themes to build 800-word passages
THEMES = [
    ("Distributed Systems", "In modern distributed computing, consensus algorithms and distributed state machines govern fault-tolerant coordination across heterogeneous nodes."),
    ("Neural Network Compilers", "Optimizing deep learning graphs requires intermediate representations capable of fusion, tiling, and memory hierarchy allocation."),
    ("Astrophysical Cosmology", "The cosmic microwave background radiation provides an observational window into the primordial fluctuations of the early universe."),
    ("Quantum Information Theory", "Quantum entanglement and superposition form the mathematical foundation for quantum computing registers and error correction."),
    ("Operating System Kernels", "Microkernels and monolithic kernels present fundamentally distinct trade-offs between context switch overhead and fault isolation."),
    ("Transformer Architectures", "Self-attention mechanisms calculate pairwise token relevance using scaled dot-product operations across multi-head projections."),
    ("Speech Synthesis Mechanics", "Acoustic codebook modeling decomposes complex continuous human speech waveforms into hierarchical discrete tokens."),
    ("Microarchitecture and Cache", "Modern superscalar processors utilize out-of-order execution engines, branch predictors, and multi-level cache hierarchies."),
    ("Database Storage Engines", "Log-structured merge trees and B-trees optimize write amplification versus point lookup latency in persistent storage."),
    ("Compiler Optimization Passes", "Static single assignment form facilitates dead code elimination, constant propagation, and loop invariant code motion."),
    ("Fluid Dynamics and Turbulence", "The Navier-Stokes equations describe the motion of viscous fluids through non-linear convective acceleration and diffusion."),
    ("Cryptography and Ciphers", "Public key cryptography relies on computationally intractable mathematical problems such as prime factorization and elliptic curves."),
    ("Network Protocol Engineering", "Congestion control algorithms in transport protocols dynamically adjust window sizes to minimize bufferbloat and latency."),
    ("Robotics and Kinematics", "Forward and inverse kinematics translate joint angles into end-effector Cartesian coordinates in high-degree-of-freedom manipulators."),
    ("Semiconductor Fabrication", "Extreme ultraviolet lithography enables sub-five-nanometer transistor gate patterning through complex reflective optics."),
    ("Reinforcement Learning", "Markov decision processes formalize agent-environment interactions where policies maximize cumulative expected discounted reward."),
    ("Memory Bandwidth Bottlenecks", "High-bandwidth memory stacks provide terabytes per second of throughput to satisfy tensor core arithmetic demand."),
    ("High-Performance Computing", "Message passing interfaces and collective communication primitives synchronize parallel domain decomposition across clusters."),
    ("Bioinformatics and Genomics", "De novo sequence assembly stitches fragmented shotgun sequencing reads into continuous chromosome representations."),
    ("Autonomous Navigation", "Simultaneous localization and mapping algorithms integrate lidar point clouds and visual odometry to estimate robot pose."),
    ("Signal Processing Theory", "The discrete Fourier transform maps discrete-time signals into orthogonal frequency domain bins via complex exponentials."),
    ("Solid-State Physics", "Bandgap engineering in semiconductor heterostructures controls electron and hole carrier confinement in quantum wells."),
    ("Statistical Mechanics", "Boltzmann distributions relate microscopic microstate energies to macroscopic thermodynamic observables like entropy."),
    ("Information Retrieval", "Inverted indices, vector embeddings, and approximate nearest neighbor graphs accelerate semantic document similarity search."),
    ("Computational Geometry", "Delaunay triangulations and Voronoi diagrams partition geometric spatial domains for mesh generation and spatial queries."),
    ("Virtualization and Hypervisors", "Hardware-assisted virtualization utilizes extended page tables and hypervisor root operation modes to isolate guest virtual machines."),
    ("Asynchronous Event Loops", "Non-blocking event demultiplexers process thousands of concurrent I/O operations through system call primitives."),
    ("Graph Neural Networks", "Message passing frameworks aggregate neighboring node representations along relational graph edges to update latent node embeddings."),
    ("Numerical Linear Algebra", "Singular value decomposition and QR factorization form stable foundations for least-squares regression and dimensionality reduction."),
    ("Computer Vision and Optics", "Epipolar geometry governs stereo vision correspondences through fundamental matrices and perspective camera projection."),
    ("Audio Codec Latency", "Neural audio codecs balance frame downsampling ratios against acoustic artifact generation in low-bitrate streaming speech."),
    ("Distributed Consensus", "Raft and Paxos algorithms guarantee safety and linearizability across distributed log replicas in the presence of network partitions."),
    ("Instruction Set Architectures", "Reduced instruction set computing prioritizes simple, uniform-length instructions with single-cycle register execution pipelines."),
    ("Tensor Core Scheduling", "Warp-level matrix multiply and accumulate instructions maximize arithmetic intensity on specialized hardware execution units."),
    ("Causal Inference Systems", "Directed acyclic graphs and structural equation models distinguish true causal mechanisms from spurious statistical correlations."),
    ("Parallel Garbage Collection", "Concurrent mark-sweep and generational collectors minimize stop-the-world pause times in memory-managed virtual machines."),
    ("Quantum Chromodynamics", "Strong nuclear interactions bind quarks and gluons inside hadrons through non-Abelian gauge field equations."),
    ("Thermal Dissipation in Silicon", "Junction temperatures and heat sink thermal resistance dictate peak sustained frequency scaling in modern accelerators."),
    ("Model Quantization Arithmetic", "Symmetric 8-bit and 4-bit integer quantization scales floating point dynamic ranges into bounded discrete integer registers."),
    ("Cloud Native Orchestration", "Declarative reconciliation controllers continuously observe and converge actual cluster state toward desired application topology.")
]

BODY_PARAGRAPHS = [
    "The systematic evaluation of accelerator architectures reveals profound interactions between compute intensity and memory bandwidth. When processing continuous sequence streams, modern graphics processing units transition through distinct computational regimes. In the prefill phase, parallel matrix multiplications achieve peak floating-point throughput by saturating arithmetic logic units. However, during iterative token generation, each forward step requires retrieving model parameters from global memory to local register files.",
    "This architectural constraint highlights the critical necessity of low-precision representations. By quantizing parameters to 8-bit and 4-bit formats, memory traffic across the interconnect is dramatically curtailed, effectively elevating arithmetic intensity. Consequently, multiple concurrent sessions can execute within the memory bus bandwidth budget, avoiding memory starvation and ensuring predictable execution latencies.",
    "Furthermore, continuous iteration-level scheduling provides an optimal mechanism for handling variable-length generation requests. Rather than locking hardware resources in coarse-grained serial queue pools, dynamic batching interleaves session progression at granular frame intervals. When a session terminates upon encountering an end-of-sequence condition, its allocated memory structures are reclaimed instantaneously, redirecting all compute cycles to active streams.",
    "The integration of dedicated neural vocoders on local accelerators eliminates cross-device bus contention. By decoding discrete codebook representations directly in local high-bandwidth memory, waveform generation proceeds concurrently with transformer inference. This decoupled, symmetric structure allows the cluster to sustain high-throughput speech synthesis across dozens of concurrent users without experiencing head-of-line blocking or latency degradation.",
    "In large-scale production environments, maintaining sub-second time to first audio while guaranteeing high cluster throughput represents the primary engineering objective. Through adaptive quantum allocation, initial frames are synthesized and emitted without delay, establishing immediate audio delivery. Subsequent frames are evaluated in batched quanta to preserve hardware efficiency and achieve unprecedented real-time generation speed."
]

def generate_800_word_prompt(theme_title: str, theme_intro: str, target_words: int = 800) -> str:
    parts = [f"Topic: {theme_title}.", theme_intro]
    while sum(len(p.split()) for p in parts) < target_words:
        for p in BODY_PARAGRAPHS:
            parts.append(p)
            if sum(len(x.split()) for x in parts) >= target_words:
                break
    words = " ".join(parts).split()
    return " ".join(words[:target_words])

def main():
    parser = argparse.ArgumentParser(description="40-User x 800-Word Maximum Stress Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to Q8_0 GGUF model")
    parser.add_argument("--q4-model", type=str, required=True, help="Path to Q4_K GGUF model")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so")
    parser.add_argument("--out-dir", type=str, default="audio_40users_800words", help="Output directory")
    parser.add_argument("--n-users", type=int, default=40, help="Number of concurrent users (default: 40)")
    parser.add_argument("--max-slots", type=int, default=20, help="Max slots per GPU (default: 20 -> 40 total)")
    parser.add_argument("--q4-threshold", type=int, default=10, help="Slots threshold to trigger Q4 (default: 10)")
    parser.add_argument("--quantum", type=int, default=2, help="Quantum frames per step (default: 2)")
    parser.add_argument("--max-steps", type=int, default=750, help="Max acoustic steps per user (default: 750)")
    args = parser.parse_args()

    print("================================================================================")
    print("        40-USER x 800-WORD MAXIMUM STRESS CLUSTER BENCHMARK")
    print("================================================================================")
    print(f"Primary Model:   {args.model} (Q8_0)")
    print(f"Surge Q4 Model:  {args.q4_model} (Q4_K, Trigger Threshold: {args.q4_threshold} slots/GPU)")
    print(f"Concurrent Users:{args.n_users} users (All admitted at t=0, Zero Queue Waiting!)")
    print(f"Slots per GPU:   {args.max_slots} (Total Cluster Slots: {args.max_slots * 2})")
    print(f"Quantum Frames:  {args.quantum} (Adaptive: 1 init, {args.quantum} steady)")
    print(f"Max Steps/User:  {args.max_steps} (~{args.max_steps*0.08:.1f}s speech per user)")
    print(f"Output Directory:{args.out_dir}")
    print("================================================================================")

    # Initialize Cluster
    cluster = DualInstanceCluster(
        model_path=args.model,
        lib_path=args.lib,
        q4_model_path=args.q4_model,
        enable_q4_burst=True,
        q4_threshold=args.q4_threshold
    )

    # Build 40 tasks of 800 words each
    print(f"\n[Preparation] Generating {args.n_users} distinct 800-word text prompts...")
    tasks = []
    total_words = 0
    for i in range(args.n_users):
        theme_title, theme_intro = THEMES[i % len(THEMES)]
        text_800 = generate_800_word_prompt(theme_title, theme_intro, target_words=800)
        n_w = len(text_800.split())
        total_words += n_w
        tasks.append(UserTask(
            id=i + 1,
            text=text_800,
            instruction="Speak clearly, naturally, and authoritatively.",
            seed=1000 + i,
            max_steps=args.max_steps
        ))

    print(f"   -> Successfully generated {len(tasks)} tasks (Total Words: {total_words:,}).")
    print(f"   -> Average words per task: {total_words / len(tasks):.1f} words.")

    # Telemetry
    telemetry = ClusterTelemetry(interval_s=0.2)
    telemetry.start()

    print(f"\n[Dispatch] Launching ALL {len(tasks)} user tasks into {args.max_slots*2} active slots at t = 0.00s...")
    t_start = time.time()
    last_heartbeat = [time.time()]

    def progress_cb(r):
        now = time.time()
        print(f"  [EOS Finished] User {r.id:02d} ({r.words:3d}w) | Audio: {r.audio_s:5.2f}s | Wall: {r.compute_wall_s:5.2f}s | TTFA: {r.ttfa_s:5.3f}s | RTF: {r.rtf:5.3f} | {r.worker}")

    results = cluster.run_workload(
        tasks,
        out_dir=args.out_dir,
        on_progress=progress_cb,
        quantum_frames=args.quantum,
        max_slots_per_gpu=args.max_slots
    )
    total_wall = time.time() - t_start

    telemetry.stop()
    telem_stats = telemetry.get_summary()

    total_audio = sum(r.audio_s for r in results)
    avg_ttfa = sum(r.ttfa_s for r in results) / len(results)
    min_ttfa = min(r.ttfa_s for r in results)
    max_ttfa = max(r.ttfa_s for r in results)
    cluster_rtf = total_wall / total_audio if total_audio > 0 else 0.0
    realtime_mult = total_audio / total_wall if total_wall > 0 else 0.0

    print("\n================================================================================")
    print("                      40-USER MAXIMUM STRESS REPORT")
    print("================================================================================")
    print(f"Total Requests:           {len(results)} users")
    print(f"Total Words Processed:    {total_words:,} words")
    print(f"Cluster Wall Time:        {total_wall:.2f} s ({total_wall/60.0:.2f} min)")
    print(f"Total Speech Synthesized: {total_audio:.2f} s ({total_audio/60.0:.2f} min)")
    print(f"Cluster Real-Time Speed:  {realtime_mult:.2f}x Real-Time (RTF: {cluster_rtf:.3f})")
    print(f"Mean TTFA:                {avg_ttfa:.3f} s")
    print(f"Min TTFA:                 {min_ttfa:.3f} s")
    print(f"Max TTFA:                 {max_ttfa:.3f} s")
    print("--------------------------------------------------------------------------------")
    print("GPU Hardware Telemetry:")
    for gpu_name, data in telem_stats.items():
        print(f"  GPU {gpu_name}: Util Mean = {data['util_mean']:.1f}% | Util Max = {data['util_max']:.1f}% | VRAM Peak = {data['mem_peak']:.1f} MiB")
    print("================================================================================")

if __name__ == "__main__":
    main()
