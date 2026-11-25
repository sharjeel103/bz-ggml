import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kaggle import execute_code, get_or_create_idle_kernel

BENCH_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_continuous_marathon_benchmark.py")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT_DIR, "python", "bz_ggml")
SRC_DIR = os.path.join(ROOT_DIR, "src")
INC_DIR = os.path.join(ROOT_DIR, "include")

with open(BENCH_SCRIPT_PATH, "r") as f:
    bench_code = f.read()

# Package python/bz_ggml, src, and include into a tar.gz in-memory
tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
    tar.add(PYTHON_DIR, arcname="python/bz_ggml")
    tar.add(SRC_DIR, arcname="src")
    tar.add(INC_DIR, arcname="include")

b64_code_tar = base64.b64encode(tar_buf.getvalue()).decode("utf-8")
b64_script = base64.b64encode(bench_code.encode("utf-8")).decode("utf-8")

REMOTE_RUNNER = f"""
import base64, io, os, sys, subprocess, glob, json, tarfile

# 1. Sync codebase
code_tar_bytes = base64.b64decode(\"\"\"{b64_code_tar}\"\"\")
with tarfile.open(fileobj=io.BytesIO(code_tar_bytes), mode="r:gz") as tar:
    tar.extractall(path="/kaggle/working/bz-ggml")

# 2. Recompile libbreeze.so
build_res = subprocess.run(
    ["cmake", "--build", "/kaggle/working/bz-ggml/build", "--target", "breeze", "-j4"],
    capture_output=True, text=True
)
if build_res.returncode != 0:
    print(">>> Recompile failed:\\n", build_res.stderr, flush=True)
    sys.exit(1)
print(">>> libbreeze.so recompiled cleanly on Kaggle.", flush=True)

# 3. Unpack benchmark script
script_bytes = base64.b64decode(\"\"\"{b64_script}\"\"\")
with open("/kaggle/working/bz-ggml/tests/verify_continuous_marathon_benchmark.py", "wb") as f:
    f.write(script_bytes)

print(">>> Benchmark suite ready. Starting continuous marathon execution...", flush=True)

# 4. Execute benchmark script and stream logs
proc = subprocess.Popen(
    ["python3", "-u", "/kaggle/working/bz-ggml/tests/verify_continuous_marathon_benchmark.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f">>> Process finished with exit code {{proc.returncode}}", flush=True)

# 5. Collect output files and output base64 bundle
results_dir = "/kaggle/working/benchmark_results"
bundle = {{}}
if os.path.exists(results_dir):
    for fname in os.listdir(results_dir):
        fpath = os.path.join(results_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                bundle[fname] = base64.b64encode(f.read()).decode("utf-8")

print("--- BEGIN_BUNDLE_JSON ---")
print(json.dumps(bundle))
print("--- END_BUNDLE_JSON ---")
"""

async def main():
    print("Connecting to Kaggle kernel and launching continuous marathon benchmark...")
    kernel_id = get_or_create_idle_kernel()
    print(f"Using kernel: {kernel_id}")
    stdout, stderr, status = await execute_code(REMOTE_RUNNER, kernel_id=kernel_id, timeout_sec=3600)
    
    # Extract bundle if present
    if "--- BEGIN_BUNDLE_JSON ---" in stdout and "--- END_BUNDLE_JSON ---" in stdout:
        part = stdout.split("--- BEGIN_BUNDLE_JSON ---")[1].split("--- END_BUNDLE_JSON ---")[0].strip()
        bundle = json.loads(part)
        
        # Save to local brain artifacts directory
        artifact_dir = "/home/sharjeel/.gemini/antigravity-cli/brain/778dd515-1d1b-4190-8ed2-8b6a2e38c033"
        local_results_dir = os.path.join(artifact_dir, "benchmark_results")
        os.makedirs(local_results_dir, exist_ok=True)
        
        print(f"\n>>> Saving {len(bundle)} retrieved benchmark artifacts to {local_results_dir}:")
        for fname, b64_data in bundle.items():
            fpath = os.path.join(local_results_dir, fname)
            with open(fpath, "wb") as f:
                f.write(base64.b64decode(b64_data))
            sz_kb = os.path.getsize(fpath) / 1024.0
            print(f"  • {fname} ({sz_kb:.1f} KB)")
    
    print("\nBenchmark Execution Complete! Status:", status)

if __name__ == "__main__":
    asyncio.run(main())
