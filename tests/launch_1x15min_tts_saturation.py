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

BENCH_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_1x15min_tts_saturation.py")
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
with open("/kaggle/working/bz-ggml/tests/verify_1x15min_tts_saturation.py", "wb") as f:
    f.write(base64.b64decode("""__B64_SCRIPT__"""))

with open("/kaggle/working/bz-ggml/tests/texts_corpus_5x20m.py", "wb") as f:
    f.write(base64.b64decode("""__B64_CORPUS__"""))

print(">>> Codebase, corpus, and benchmark script synchronized on Kaggle.", flush=True)
print(">>> Starting 1x 15-min Isolated TTS Benchmark (CFG=1.0)...", flush=True)

# 3. Execute benchmark script and stream live logs
proc = subprocess.Popen(
    ["python3", "-u", "/kaggle/working/bz-ggml/tests/verify_1x15min_tts_saturation.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f"\\n>>> Benchmark process exited with returncode: {proc.returncode}", flush=True)

# 4. Package benchmark results for retrieval
results_dir = "/kaggle/working/benchmark_results"
manifest = {}
if os.path.exists(results_dir):
    files_to_bundle = []
    # Collect summary, telemetry, log, png dashboards, and WAV audio
    for pattern in ["summary_1x15m.json", "telemetry_1x15m.csv", "benchmark_1x15m_live.log", "*.png", "audio/req_1x15m_cfg1.wav"]:
        matches = glob.glob(os.path.join(results_dir, pattern))
        for m in matches:
            if os.path.isfile(m):
                rel = os.path.relpath(m, results_dir)
                files_to_bundle.append((m, rel))

    out_tar_buf = io.BytesIO()
    with tarfile.open(fileobj=out_tar_buf, mode="w:gz") as out_tar:
        for full_p, rel_p in files_to_bundle:
            out_tar.add(full_p, arcname=rel_p)

    manifest["results_tar_b64"] = base64.b64encode(out_tar_buf.getvalue()).decode("utf-8")
    manifest["file_count"] = len(files_to_bundle)
    manifest["total_bytes"] = len(out_tar_buf.getvalue())

print("<<<RESULTS_BUNDLE_BEGIN>>>")
print(json.dumps(manifest))
print("<<<RESULTS_BUNDLE_END>>>")
'''.replace("__B64_CODE_TAR__", b64_code_tar).replace("__B64_SCRIPT__", b64_script).replace("__B64_CORPUS__", b64_corpus)

async def main():
    print("=" * 80)
    print(">>> PRE-FLIGHT: Restarting Kaggle Jupyter kernel to clear GPU VRAM to 0 MiB...")
    print("=" * 80)
    restarted = restart_kernel()
    if restarted:
        print(">>> Kernel restart initiated. Waiting 5s for clean kernel...")
        time.sleep(5)
    else:
        print("[-] Warning: Kernel restart POST failed, proceeding anyway...")

    kernel_id = get_or_create_idle_kernel()
    print(f">>> Connecting to idle kernel: {kernel_id}")
    print(f">>> Launching 1x 15-Minute Isolated TTS Benchmark (timeout: 1800s)...")
    print("=" * 80)

    t0 = time.time()
    stdout_out, stderr_out, status = await execute_code(REMOTE_RUNNER, kernel_id=kernel_id, timeout_sec=1800)
    wall = time.time() - t0
    print(f"\n>>> Remote execution finished with status '{status}' in {wall:.1f}s.")

    # Extract results bundle
    if "<<<RESULTS_BUNDLE_BEGIN>>>" in stdout_out and "<<<RESULTS_BUNDLE_END>>>" in stdout_out:
        b_start = stdout_out.index("<<<RESULTS_BUNDLE_BEGIN>>>") + len("<<<RESULTS_BUNDLE_BEGIN>>>")
        b_end = stdout_out.index("<<<RESULTS_BUNDLE_END>>>")
        bundle_str = stdout_out[b_start:b_end].strip()
        manifest = json.loads(bundle_str)
        b64_tar = manifest.get("results_tar_b64", "")
        if b64_tar:
            tar_bytes = base64.b64decode(b64_tar)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as res_tar:
                res_tar.extractall(path=LOCAL_RESULTS_DIR)
            print(f">>> Extracted {manifest.get('file_count', 0)} result files ({manifest.get('total_bytes', 0) / 1024:.1f} KB) to:")
            print(f"    {LOCAL_RESULTS_DIR}")
    else:
        print("[-] Could not find results bundle marker in remote stdout.")

if __name__ == "__main__":
    asyncio.run(main())
