import os
import subprocess
import time
from typing import Dict, List, Any

class TelemetryProfiler:
    """Continuous 200ms GPU profiler querying nvidia-smi."""

    def __init__(self, log_csv_path: str = "gpu_telemetry.csv", sample_interval_ms: int = 200):
        self.csv_path = log_csv_path
        self.sample_interval_ms = sample_interval_ms
        self.proc = None

    def start(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

        cmd = (
            f"nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used "
            f"--format=csv,nounits,noheader -lms {self.sample_interval_ms} > {self.csv_path}"
        )
        self.proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
        time.sleep(0.5)

    def stop(self) -> Dict[str, Any]:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except Exception:
                self.proc.kill()
            self.proc = None

        g0_utils, g0_mems, g0_pwrs = [], [], []
        g1_utils, g1_mems, g1_pwrs = [], [], []

        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r") as f:
                for line in f:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        try:
                            idx = int(parts[1])
                            u_gpu = float(parts[2])
                            pwr = float(parts[4])
                            mem = float(parts[5])
                            if idx == 0:
                                g0_utils.append(u_gpu)
                                g0_pwrs.append(pwr)
                                g0_mems.append(mem)
                            elif idx == 1:
                                g1_utils.append(u_gpu)
                                g1_pwrs.append(pwr)
                                g1_mems.append(mem)
                        except ValueError:
                            continue

        return {
            "gpu0": {
                "mean_utilization_pct": round(sum(g0_utils)/len(g0_utils), 2) if g0_utils else 0.0,
                "mean_power_w": round(sum(g0_pwrs)/len(g0_pwrs), 2) if g0_pwrs else 0.0,
                "peak_vram_mb": round(max(g0_mems), 1) if g0_mems else 0.0,
                "samples": len(g0_utils)
            },
            "gpu1": {
                "mean_utilization_pct": round(sum(g1_utils)/len(g1_utils), 2) if g1_utils else 0.0,
                "mean_power_w": round(sum(g1_pwrs)/len(g1_pwrs), 2) if g1_pwrs else 0.0,
                "peak_vram_mb": round(max(g1_mems), 1) if g1_mems else 0.0,
                "samples": len(g1_utils)
            }
        }

def calculate_jains_fairness(results: List[Dict[str, Any]]) -> float:
    rates = [(r["audio_s"] / r["turnaround_s"]) if r.get("turnaround_s", 0) > 0 else 0.0 for r in results]
    if not rates or sum(x**2 for x in rates) == 0:
        return 0.0
    return (sum(rates) ** 2) / (len(rates) * sum(x**2 for x in rates))
