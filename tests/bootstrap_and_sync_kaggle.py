import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kaggle import execute_code, get_or_create_idle_kernel

TAR_PATH = "/tmp/bz_src.tar.gz"
if not os.path.exists(TAR_PATH):
    print(f"Error: {TAR_PATH} does not exist.")
    sys.exit(1)

with open(TAR_PATH, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

print(f"Loaded {TAR_PATH}: {len(b64_data)/(1024*1024):.2f} MB base64 payload.")

REMOTE_SCRIPT = f"""
import os, sys, time, subprocess, base64

t0 = time.time()
print("=" * 70)
print(">>> KAGGLE REMOTE BOOTSTRAP & SYNC")
print("=" * 70)

# 1. Symlink CUDA driver
print("Step 1: Symlink CUDA driver...", flush=True)
subprocess.run("ln -sf /usr/local/nvidia/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && ldconfig", shell=True)

# 2. Extract uploaded code into /kaggle/working/bz-ggml
print("Step 2: Unpacking uploaded codebase...", flush=True)
os.makedirs("/kaggle/working/bz-ggml", exist_ok=True)
tar_b64 = \"\"\"{b64_data}\"\"\"
tar_bytes = base64.b64decode(tar_b64)
with open("/kaggle/working/bz_src.tar.gz", "wb") as f:
    f.write(tar_bytes)

subprocess.run("tar -xzf /kaggle/working/bz_src.tar.gz -C /kaggle/working/bz-ggml/", shell=True, check=True)
print(f"Extracted to /kaggle/working/bz-ggml ({{len(tar_bytes)/(1024*1024):.2f}} MB)")

# 3. Handle C++ shared library
print("Step 3: Checking C++ library and CUDA kernels...", flush=True)
os.makedirs("/kaggle/working/bz-ggml/build", exist_ok=True)
breeze_so = "/kaggle/working/bz-ggml/build/libbreeze.so"

need_compile = True
if not os.path.exists(breeze_so):
    bin_tar = "/kaggle/working/breeze-cuda-sm75.tar.gz"
    BIN_URL = "https://github.com/sharjeel103/bz-ggml/raw/prebuilt-t4/breeze-cuda-sm75.tar.gz"
    print(f"Downloading prebuilt binaries from {{BIN_URL}}...", flush=True)
    res = subprocess.run(f"wget -q -c {{BIN_URL}} -O {{bin_tar}}", shell=True)
    if res.returncode == 0:
        subprocess.run(f"tar -xzf {{bin_tar}} -C /kaggle/working/bz-ggml/", shell=True)
        subprocess.run("cp -P /kaggle/working/bz-ggml/breeze-cuda-sm75/lib/* /kaggle/working/bz-ggml/build/", shell=True)
        # Check if breeze_vocoder_decode_tiled exists
        chk = subprocess.run(f"nm -D {{breeze_so}} | grep breeze_vocoder_decode_tiled", shell=True, capture_output=True, text=True)
        if "breeze_vocoder_decode_tiled" in chk.stdout:
            print("Prebuilt binary has breeze_vocoder_decode_tiled! Compilation not required.", flush=True)
            need_compile = False
        else:
            print("Prebuilt binary missing breeze_vocoder_decode_tiled symbol. Will compile from source.", flush=True)

if need_compile:
    print("Compiling libbreeze.so with CUDA sm_75 from source...", flush=True)
    os.chdir("/kaggle/working/bz-ggml")
    subprocess.run("cmake -B build -DBREEZE_CUDA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release", shell=True, check=True)
    subprocess.run("cmake --build build --config Release -j$(nproc)", shell=True, check=True)

# Link GGML libraries into system library path
subprocess.run("ln -sf /kaggle/working/bz-ggml/build/third_party/ggml/src/*.so* /usr/lib/x86_64-linux-gnu/", shell=True)
subprocess.run("ln -sf /kaggle/working/bz-ggml/build/third_party/ggml/src/ggml-cuda/*.so* /usr/lib/x86_64-linux-gnu/", shell=True)
subprocess.run("ln -sf /kaggle/working/bz-ggml/build/*.so* /usr/lib/x86_64-linux-gnu/", shell=True)
subprocess.run("ldconfig", shell=True)
print(f"libbreeze.so ready: {{os.path.exists(breeze_so)}}", flush=True)

# 4. Install python dependencies
print("Step 4: Installing pinned python packages...", flush=True)
subprocess.run("pip install -q 'av<18' faster-whisper soundfile resemble-enhance gradio matplotlib pandas", shell=True)

# 5. Download model (Q8_0 for maximum VRAM efficiency and 150-chunk pool headroom)
print("Step 5: Verifying model file...", flush=True)
os.makedirs("/kaggle/working/models", exist_ok=True)
q8_path = "/kaggle/working/models/breeze-tts-2-q8_0.gguf"
if not os.path.exists(q8_path):
    print("Downloading breeze-tts-2-q8_0.gguf (~3.9 GB) from Hugging Face...", flush=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="HoppouAI/Breeze-TTS-2.cpp", filename="breeze-tts-2-q8_0.gguf", local_dir="/kaggle/working/models")
print(f"Model ready: {{q8_path}} ({{os.path.getsize(q8_path)/(1024*1024):.1f}} MB)", flush=True)

print(f"Bootstrap and setup finished in {{time.time() - t0:.1f}}s!")
"""

async def main():
    print("Connecting to Kaggle kernel and launching bootstrap...")
    kernel_id = get_or_create_idle_kernel()
    print(f"Using kernel: {kernel_id}")
    stdout, stderr, status = await execute_code(REMOTE_SCRIPT, kernel_id=kernel_id, timeout_sec=1800)
    print("Status:", status)

if __name__ == "__main__":
    asyncio.run(main())
