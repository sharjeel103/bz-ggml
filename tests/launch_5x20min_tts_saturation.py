import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kaggle import execute_code, get_or_create_idle_kernel, restart_kernel

BENCH_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_5x20min_tts_saturation.py")
CORPUS_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts_corpus_5x20m.py")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT_DIR, "python", "bz_ggml")
LOCAL_RESULTS_DIR = "/home/sharjeel/.gemini/antigravity-cli/brain/778dd515-1d1b-4190-8ed2-8b6a2e38c033/benchmark_results"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)

with open(BENCH_SCRIPT_PATH, "r") as f:
    bench_code = f.read()

with open(CORPUS_SCRIPT_PATH, "r") as f:
    corpus_code = f.read()

# Package python/bz_ggml into in-memory tar.gz
tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
    tar.add(PYTHON_DIR, arcname="python/bz_ggml")

b64_code_tar = base64.b64encode(tar_buf.getvalue()).decode("utf-8")
b64_script = base64.b64encode(bench_code.encode("utf-8")).decode("utf-8")
b64_corpus = base64.b64encode(corpus_code.encode("utf-8")).decode("utf-8")

REMOTE_RUNNER = '''
import base64, io, os, sys, subprocess, glob, json, tarfile

# 1. Sync latest python/bz_ggml code
code_tar_bytes = base64.b64decode("""__B64_CODE_TAR__""")
with tarfile.open(fileobj=io.BytesIO(code_tar_bytes), mode="r:gz") as tar:
    tar.extractall(path="/kaggle/working/bz-ggml")

# 2. Write benchmark and corpus scripts
os.makedirs("/kaggle/working/bz-ggml/tests", exist_ok=True)
with open("/kaggle/working/bz-ggml/tests/verify_5x20min_tts_saturation.py", "wb") as f:
    f.write(base64.b64decode("""__B64_SCRIPT__"""))

with open("/kaggle/working/bz-ggml/tests/texts_corpus_5x20m.py", "wb") as f:
    f.write(base64.b64decode("""__B64_CORPUS__"""))

print(">>> Codebase, corpus, and benchmark script synchronized on Kaggle.", flush=True)
print(">>> Starting 5x 20-min TTS Saturation Benchmark execution...", flush=True)

# 3. Execute benchmark script and stream live logs
proc = subprocess.Popen(
    ["python3", "-u", "/kaggle/working/bz-ggml/tests/verify_5x20min_tts_saturation.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in iter(proc.stdout.readline, ""):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f">>> Benchmark subprocess finished with exit code {proc.returncode}", flush=True)

# Report saved audio files in /kaggle/working/benchmark_results/audio
audio_files = glob.glob("/kaggle/working/benchmark_results/audio/*.wav")
print(f"\\n>>> Verified Saved Audio Files on Kaggle ({len(audio_files)} files):", flush=True)
for af in sorted(audio_files):
    size_mb = os.path.getsize(af) / (1024 * 1024)
    print(f"    - {af} ({size_mb:.2f} MB)", flush=True)

# 4. Package benchmark metrics and plots for local retrieval
res_files = [
    "/kaggle/working/benchmark_results/summary_5x20m.json",
    "/kaggle/working/benchmark_results/telemetry_5x20m.csv",
    "/kaggle/working/benchmark_results/benchmark_live.log",
    "/kaggle/working/benchmark_results/gpu_compute_5x20m.png",
    "/kaggle/working/benchmark_results/vram_saturation_5x20m.png",
    "/kaggle/working/benchmark_results/tokens_timeline_5x20m.png"
]
res_tar_buf = io.BytesIO()
with tarfile.open(fileobj=res_tar_buf, mode="w:gz") as tar:
    for rf in res_files:
        if os.path.exists(rf):
            tar.add(rf, arcname=os.path.basename(rf))

b64_res = base64.b64encode(res_tar_buf.getvalue()).decode("utf-8")
print(f"__BENCH_RESULTS_TAR_START__{b64_res}__BENCH_RESULTS_TAR_END__", flush=True)
'''.replace("__B64_CODE_TAR__", b64_code_tar).replace("__B64_SCRIPT__", b64_script).replace("__B64_CORPUS__", b64_corpus)

async def main():
    print("Connecting to Kaggle kernel and launching 5x 20-min TTS saturation benchmark...")
    kernel_id = get_or_create_idle_kernel()
    print(f"Using kernel: {kernel_id}")
    print("Restarting kernel to ensure 100% clean GPUs (0 MiB used)...")
    restart_kernel(kernel_id)
    time.sleep(3)

    # Set execution timeout to 2400s (40 mins)
    stdout_str, stderr_str, status = await execute_code(REMOTE_RUNNER, kernel_id=kernel_id, timeout_sec=2400)

    # Extract results if captured in stdout
    if "__BENCH_RESULTS_TAR_START__" in stdout_str and "__BENCH_RESULTS_TAR_END__" in stdout_str:
        start_idx = stdout_str.index("__BENCH_RESULTS_TAR_START__") + len("__BENCH_RESULTS_TAR_START__")
        end_idx = stdout_str.index("__BENCH_RESULTS_TAR_END__")
        b64_res = stdout_str[start_idx:end_idx].strip()
        tar_bytes = base64.b64decode(b64_res)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(path=LOCAL_RESULTS_DIR)
        print(f"\n>>> Successfully downloaded all benchmark artifacts to {LOCAL_RESULTS_DIR}")
        for f in os.listdir(LOCAL_RESULTS_DIR):
            if "5x20m" in f or "benchmark_live" in f:
                print(f"    - {f}")
    else:
        print("Note: Artifact tar marker not found in stdout buffer.")

if __name__ == "__main__":
    asyncio.run(main())
