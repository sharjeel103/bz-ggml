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

BENCH_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_1x15min_fp16_tts.py")
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
import base64, io, os, sys, subprocess, glob, json, tarfile, time

print("=" * 80, flush=True)
print(">>> STEP 1: Setting up codebase and prebuilt CUDA binaries on Kaggle...", flush=True)
print("=" * 80, flush=True)

os.makedirs("/kaggle/working/bz-ggml/build", exist_ok=True)
os.makedirs("/kaggle/working/bz-ggml/tests", exist_ok=True)
os.makedirs("/kaggle/working/models", exist_ok=True)

# 1. Sync latest python/bz_ggml code
code_tar_bytes = base64.b64decode("""__B64_CODE_TAR__""")
with tarfile.open(fileobj=io.BytesIO(code_tar_bytes), mode="r:gz") as tar:
    tar.extractall(path="/kaggle/working/bz-ggml")

# 2. Write benchmark and corpus scripts
with open("/kaggle/working/bz-ggml/tests/verify_1x15min_fp16_tts.py", "wb") as f:
    f.write(base64.b64decode("""__B64_SCRIPT__"""))

with open("/kaggle/working/bz-ggml/tests/texts_corpus_5x20m.py", "wb") as f:
    f.write(base64.b64decode("""__B64_CORPUS__"""))

# 3. Setup prebuilt libbreeze.so
breeze_so = "/kaggle/working/bz-ggml/build/libbreeze.so"
if not os.path.exists(breeze_so):
    bin_tar = "/kaggle/working/breeze-cuda-sm75.tar.gz"
    if not os.path.exists(bin_tar):
        bin_url = "https://github.com/sharjeel103/bz-ggml/raw/prebuilt-t4/breeze-cuda-sm75.tar.gz"
        print(f"Downloading prebuilt binaries from {bin_url}...", flush=True)
        subprocess.run(f"wget -q -c {bin_url} -O {bin_tar}", shell=True, check=True)
    subprocess.run(f"tar -xzf {bin_tar} -C /kaggle/working/", shell=True, check=True)
    subprocess.run("cp -P /kaggle/working/breeze-cuda-sm75/lib/* /kaggle/working/bz-ggml/build/", shell=True, check=True)
    subprocess.run("ln -sf /kaggle/working/bz-ggml/build/*.so* /usr/lib/x86_64-linux-gnu/", shell=True)
    subprocess.run("ldconfig", shell=True)

print(f"libbreeze.so ready at {breeze_so}: {os.path.exists(breeze_so)}", flush=True)

# 4. Install missing dependencies
print("\\n>>> STEP 2: Verifying python packages...", flush=True)
subprocess.run("pip install -q 'av<18'", shell=True)

# 5. Download unquantized FP16 model (breeze-tts-2-f16.gguf)
print("\\n>>> STEP 3: Verifying unquantized FP16 model (breeze-tts-2-f16.gguf)...", flush=True)
f16_path = "/kaggle/working/models/breeze-tts-2-f16.gguf"
if not os.path.exists(f16_path) or os.path.getsize(f16_path) < 1000000000:
    print("Downloading breeze-tts-2-f16.gguf (~6.8 GB) from HoppouAI/Breeze-TTS-2.cpp...", flush=True)
    t_dl0 = time.time()
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="HoppouAI/Breeze-TTS-2.cpp", filename="breeze-tts-2-f16.gguf", local_dir="/kaggle/working/models")
    print(f"FP16 Model downloaded in {time.time() - t_dl0:.1f}s!", flush=True)

print(f"FP16 Model ready: {f16_path} ({os.path.getsize(f16_path) / (1024*1024):.1f} MB)", flush=True)

print("\\n" + "=" * 80, flush=True)
print(">>> STEP 4: Launching 1x 15-Minute Maximum Quality FP16 Benchmark (CFG = 5.0)...", flush=True)
print("=" * 80, flush=True)

# 6. Execute benchmark script and stream live logs
proc = subprocess.Popen(
    ["python3", "-u", "/kaggle/working/bz-ggml/tests/verify_1x15min_fp16_tts.py"],
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

# 7. Package benchmark results for retrieval
results_dir = "/kaggle/working/benchmark_results"
manifest = {}
if os.path.exists(results_dir):
    files_to_bundle = []
    for pattern in ["summary_1x15m_fp16.json", "telemetry_1x15m_fp16.csv", "benchmark_1x15m_fp16_live.log", "*.png", "audio/req_1x15m_fp16_cfg5.wav"]:
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
    kernel_id = get_or_create_idle_kernel()
    print("=" * 80)
    print(f">>> Connecting to idle kernel: {kernel_id}")
    print(f">>> Launching Maximum Quality FP16 TTS Benchmark (timeout: 2400s)...")
    print("=" * 80)

    t0 = time.time()
    stdout_out, stderr_out, status = await execute_code(REMOTE_RUNNER, kernel_id=kernel_id, timeout_sec=2400)
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
            print(f">>> Extracted {manifest.get('file_count', 0)} result files ({manifest.get('total_bytes', 0) / (1024*1024):.2f} MB) to:")
            print(f"    {LOCAL_RESULTS_DIR}")
    else:
        print("[-] Notice: Result bundle marker was split in streaming buffer; retrieval via direct pull script will follow if needed.")

if __name__ == "__main__":
    asyncio.run(main())
