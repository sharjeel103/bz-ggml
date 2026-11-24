import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kaggle import execute_code, get_or_create_idle_kernel

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_enhancer_and_asr_exact_vram.py")

with open(SCRIPT_PATH, "r") as f:
    script_code = f.read()

b64_script = base64.b64encode(script_code.encode("utf-8")).decode("utf-8")

REMOTE_RUNNER = f"""
import base64, subprocess, sys, os

# Write script to Kaggle disk
script_bytes = base64.b64decode(\"\"\"{b64_script}\"\"\")
target_path = "/kaggle/working/profile_enhancer_and_asr_exact_vram.py"
with open(target_path, "wb") as f:
    f.write(script_bytes)

print(">>> Profiler script written to Kaggle. Starting isolated execution...", flush=True)

proc = subprocess.Popen(
    ["python3", "-u", target_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f"\\n>>> Profiler finished with exit code {{proc.returncode}}", flush=True)
"""

async def main():
    print("Connecting to Kaggle kernel and launching isolated VRAM profiler...")
    kernel_id = get_or_create_idle_kernel()
    print(f"Using kernel: {kernel_id}")
    stdout, stderr, status = await execute_code(REMOTE_RUNNER, kernel_id=kernel_id, timeout_sec=1200)
    print("\nBenchmark Execution Complete! Status:", status)

if __name__ == "__main__":
    asyncio.run(main())
