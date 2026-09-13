#!/usr/bin/env python3
"""
40-Request Instant Burst Benchmark for bz-ggml Dual-Instance Architecture.
Dispatches 40 real-world sentences simultaneously at t = 0.00s across Dual GPUs.
"""

import argparse
import json
import os
import sys
import time

# Add parent dir to sys.path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bz_ggml import DualInstanceCluster, UserTask, TelemetryProfiler
from bz_ggml.telemetry import calculate_jains_fairness

REQUESTS_40 = [
    {"id": 1,  "text": "Your order has been confirmed and is currently being packed for express delivery this afternoon."},
    {"id": 2,  "text": "The system update completed successfully without any errors, and all security patches have been safely installed."},
    {"id": 3,  "text": "Good morning! Please let me know how I can assist you with your schedule and appointments today."},
    {"id": 4,  "text": "Flight seven twelve to San Francisco is now boarding at terminal two, gate twenty-four."},
    {"id": 5,  "text": "The project planning meeting has been rescheduled to Thursday at three in the afternoon, conference room four."},
    {"id": 6,  "text": "Your payment was processed successfully, and an itemized digital receipt has been sent to your primary email."},
    {"id": 7,  "text": "The air quality index is currently moderate across the valley with light northwesterly winds throughout the morning."},
    {"id": 8,  "text": "All cluster network services are running normally with optimal bandwidth and zero packet loss detected."},
    {"id": 9,  "text": "Please remember to submit your weekly engineering progress report before five this evening for team review."},
    {"id": 10, "text": "Your ride has arrived outside the main hotel lobby. The silver vehicle license plate is five alpha seven."},
    {"id": 11, "text": "Today will be mostly clear and sunny with mild afternoon temperatures reaching seventy-four degrees across the city."},
    {"id": 12, "text": "The conference keynote begins in ten minutes in the primary auditorium on floor three, open to all attendees."},
    {"id": 13, "text": "Your prescription order is ready for pickup at the neighborhood pharmacy counter on Maple Avenue."},
    {"id": 14, "text": "The express commuter train to central station will depart from platform four in exactly six minutes."},
    {"id": 15, "text": "A new firmware update is available for your smart display. Please ensure a stable Wi-Fi connection to proceed."},
    {"id": 16, "text": "Your checking account balance has been updated following the recent automated monthly savings transfer."},
    {"id": 17, "text": "The university library will be closing in fifteen minutes. Please bring all borrowed materials to the front circulation desk."},
    {"id": 18, "text": "Traffic on the interstate highway is moving smoothly with an estimated total travel time of twenty-two minutes."},
    {"id": 19, "text": "Welcome to the national science center. Guided audio tours commence every hour on the hour at the main rotunda."},
    {"id": 20, "text": "Your table reservation for four guests at Bistro Bella has been confirmed for eight tonight on the patio."},
    {"id": 21, "text": "The morning courier package has been safely delivered to the front reception desk for your immediate collection."},
    {"id": 22, "text": "Routine server infrastructure maintenance is scheduled for tonight at midnight and will last approximately one hour."},
    {"id": 23, "text": "Temperatures will drop noticeably tonight under clear starry skies with a gentle autumn breeze from the north."},
    {"id": 24, "text": "Your international flight check-in is complete, and your digital boarding passes have been synchronized to your phone."},
    {"id": 25, "text": "The live technical webinar on distributed computing architectures will begin promptly at noon Eastern Standard Time."},
    {"id": 26, "text": "Security notification: a new login was detected from a personal laptop in Chicago, Illinois. Please verify your identity."},
    {"id": 27, "text": "The passenger elevator on the north wing is currently undergoing maintenance and will reopen at two this afternoon."},
    {"id": 28, "text": "Your premium software subscription has been renewed successfully, unlocking continuous priority access to all cloud tools."},
    {"id": 29, "text": "Passengers traveling to terminal B should proceed to shuttle stop three for immediate baggage transfer."},
    {"id": 30, "text": "The downtown business shuttle departs every fifteen minutes from the central transit plaza near the historic clock tower."},
    {"id": 31, "text": "A temporary authorization code has been dispatched to your mobile phone number via secure text messaging."},
    {"id": 32, "text": "The resident fitness facility will remain open until eleven tonight for all registered hotel and club members."},
    {"id": 33, "text": "Local traffic monitors report minor road construction delays near the east river crossing during evening peak hours."},
    {"id": 34, "text": "Your analytical quarterly summary report has finished generating and is now available for download on the management portal."},
    {"id": 35, "text": "The interactive workshop on modern deep learning frameworks begins at ten sharp in computer laboratory C."},
    {"id": 36, "text": "Thank you for visiting our technology showroom today. Please take your complimentary catalog and have a wonderful day."},
    {"id": 37, "text": "Tomorrow's weather forecast calls for brief morning showers followed by pleasant sunshine and light southerly breezes."},
    {"id": 38, "text": "Your consultation appointment with Doctor Reynolds has been confirmed for Tuesday morning at ten thirty."},
    {"id": 39, "text": "The production cluster deployment completed without incident, and all containerized microservices report healthy operational status."},
    {"id": 40, "text": "All pending banking transactions have cleared, and your comprehensive monthly financial statement is now available to view."}
]

def main():
    parser = argparse.ArgumentParser(description="bz-ggml 40-Request Burst Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to breeze-tts-2-q8_0.gguf model")
    parser.add_argument("--lib", type=str, default=None, help="Path to libbreeze.so (optional)")
    parser.add_argument("--out-dir", type=str, default="dual_gen_40burst_audio", help="Output directory for WAVs")
    parser.add_argument("--profile-csv", type=str, default="dual_gen_40burst_profile.csv", help="CSV path for 200ms telemetry")
    parser.add_argument("--results-json", type=str, default="dual_gen_40burst_results.json", help="JSON summary output")
    args = parser.parse_args()

    print("=" * 85)
    print("STARTING 40-REQUEST INSTANT BURST BENCHMARK (DUAL TESLA T4s)")
    print("  GPU 0: Text Encoder + Generator Instance A (Backbone + 15-step Depth)")
    print("  GPU 1: Generator Instance B (Backbone + 15-step Depth) + Dedicated Streaming Vocoder")
    print("  Zero Intra-Frame PCIe Ping-Pong | Sub-250ms Warm TTFA | 8-Frame Batched Chunks")
    print("=" * 85)

    # 1. Start Telemetry
    profiler = TelemetryProfiler(args.profile_csv, sample_interval_ms=200)
    profiler.start()

    # 2. Start Cluster
    cluster = DualInstanceCluster(args.model, args.lib)

    tasks = [UserTask(id=r["id"], text=r["text"]) for r in REQUESTS_40]

    def on_progress(res):
        print(f" -> [FINISH {res.id:02d}/40] {res.worker}: Audio={res.audio_s:4.2f}s, Wall={res.compute_wall_s:4.2f}s, QueueWait={res.queue_wait_s:5.2f}s, TTFA={res.ttfa_s:5.3f}s, RTF={res.rtf:5.3f}")

    print(f"\n[BURST DISPATCH] Pushing all 40 requests simultaneously at t = 0.00s...")
    t_start = time.time()
    results = cluster.run_workload(tasks, out_dir=args.out_dir, on_progress=on_progress)
    total_wall_s = time.time() - t_start

    # 3. Stop Telemetry
    telemetry_data = profiler.stop()
    cluster.close()

    # 4. Summary & Scorecard
    total_audio_s = sum(r.audio_s for r in results)
    avg_queue_wait = sum(r.queue_wait_s for r in results) / len(results)
    avg_ttfa = sum(r.ttfa_s for r in results) / len(results)
    agg_throughput = total_audio_s / total_wall_s
    results_dicts = [r.__dict__ for r in results]
    jain_index = calculate_jains_fairness(results_dicts)

    summary = {
        "architecture": "Dual-Instance Generator (A on GPU0, B on GPU1) + Streaming Vocoder on GPU1",
        "workload": "40-Request Instant Burst (t=0.00s)",
        "total_requests": len(results),
        "total_cluster_wall_s": round(total_wall_s, 2),
        "total_audio_s": round(total_audio_s, 2),
        "aggregate_throughput_x_realtime": round(agg_throughput, 2),
        "aggregate_cluster_rtf": round(total_wall_s / total_audio_s, 3) if total_audio_s > 0 else 0.0,
        "avg_queue_wait_s": round(avg_queue_wait, 2),
        "avg_user_perceived_ttfa_s": round(avg_ttfa, 3),
        "jains_fairness_index": round(jain_index, 3),
        "telemetry": telemetry_data,
        "results": results_dicts
    }

    with open(args.results_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("40-REQUEST BURST LOAD BENCHMARK COMPLETE!")
    print("=" * 85)
    print(f"Total Cluster Wall Time    : {total_wall_s:.2f} s")
    print(f"Total Speech Synthesized   : {total_audio_s:.2f} s ({total_audio_s/60.0:.2f} min)")
    print(f"Cluster Aggregate Throughput: {agg_throughput:.2f}x Real-Time (Cluster RTF: {summary['aggregate_cluster_rtf']})")
    print(f"Average Queue Wait Time    : {avg_queue_wait:.2f} s")
    print(f"Average User TTFA          : {avg_ttfa:.3f} s")
    print(f"Jain's Fairness Index      : {jain_index:.3f}")
    print(f"GPU 0 (Generator A)        : Mean Util={telemetry_data['gpu0']['mean_utilization_pct']}%, Peak VRAM={telemetry_data['gpu0']['peak_vram_mb']} MB")
    print(f"GPU 1 (Gen B + Vocoder)    : Mean Util={telemetry_data['gpu1']['mean_utilization_pct']}%, Peak VRAM={telemetry_data['gpu1']['peak_vram_mb']} MB")
    print("=" * 85)
    print(f"{'Req ID':<7} | {'Words':<6} | {'Audio (s)':<10} | {'QueueWait':<10} | {'ComputeWall':<11} | {'User TTFA':<10} | {'RTF':<6} | {'Worker'}")
    print("-" * 85)
    for r in results:
        print(f"User {r.id:02d} | {r.words:5d}w | {r.audio_s:9.2f} s | {r.queue_wait_s:9.2f} s | {r.compute_wall_s:10.2f} s | {r.ttfa_s:9.3f} s | {r.rtf:6.3f} | {r.worker}")
    print("=" * 85)

if __name__ == "__main__":
    main()
