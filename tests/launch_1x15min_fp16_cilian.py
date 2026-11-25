"""
Remote orchestrator launcher for:
  1. Bootstrapping fresh Kaggle Dual-T4 session (/k/373698562/) with latest bug-fixed python/bz_ggml,
     prebuilt libbreeze.so (sm_75), faster-whisper, resemble-enhance, and breeze-tts-2-f16.gguf.
  2. Uploading cilian_9sec.mp3 and running Phase 1 Voice Registration (register_cilian_voice.py).
  3. Running Phase 2 Full 15-min Cloned Cilian FP16 TTS (verify_1x15min_fp16_cilian.py).
  4. Safely pulling all artifacts (cilian.breeze, telemetry CSV, JSON, PNG, and BOTH raw & mastered WAVs via FLAC)
     into the local brain artifact directory so the user can listen immediately.
"""

import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import time
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_kaggle import CTX, PROXY_URL, _require_proxy_url, execute_code, get_or_create_idle_kernel, purge_extra_kernels, restart_kernel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT_DIR, "python", "bz_ggml")
CILIAN_MP3_PATH = "/home/sharjeel/Downloads/Ai/History/Wager/characters/Character_voices/cilian_9sec.mp3"
LOCAL_RESULTS_DIR = "/home/sharjeel/.gemini/antigravity-cli/brain/778dd515-1d1b-4190-8ed2-8b6a2e38c033/benchmark_results"
LOCAL_AUDIO_DIR = os.path.join(LOCAL_RESULTS_DIR, "audio")
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_AUDIO_DIR, exist_ok=True)


def _read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def pull_remote_file(remote_path: str, local_path: str, kernel_id: str):
    """Downloads a remote file via Jupyter HTTPS /files endpoint (falling back to WebSocket slices)."""
    import hashlib
    import urllib.parse
    import urllib.request

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if remote_path.startswith("/kaggle/working/"):
        rel = remote_path[len("/kaggle/working/") :]
        url = PROXY_URL.rstrip("/") + "/files/" + urllib.parse.quote(rel)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=CTX, timeout=300) as resp:
                total = 0
                with open(local_path, "wb") as out_f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        total += len(chunk)
            if total > 0:
                print(f">>> Pulled {os.path.basename(local_path)} ({total/(1024*1024):.2f} MB via HTTP /files) -> {local_path}")
                return True
        except Exception as e:
            print(f"[!] HTTP /files download fallback for {rel}: {e}")

    meta_code = f"""
import hashlib, json, os
p = {remote_path!r}
if not os.path.exists(p):
    print("META:" + json.dumps({{"exists": False}}))
else:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    print("META:" + json.dumps({{"exists": True, "size": os.path.getsize(p), "sha256": h.hexdigest()}}))
"""
    stdout, _, _ = await execute_code(meta_code, kernel_id=kernel_id, timeout_sec=120)
    meta = None
    for line in stdout.splitlines():
        if line.startswith("META:"):
            meta = json.loads(line[5:])
            break
    if not meta or not meta.get("exists"):
        print(f"[-] Remote file missing: {remote_path}")
        return False

    total_size = int(meta["size"])
    expected_sha = meta["sha256"]
    slice_bytes = 2 * 1024 * 1024
    h_local = hashlib.sha256()

    with open(local_path, "wb") as out_f:
        offset = 0
        idx = 0
        while offset < total_size:
            read_code = f"""
import base64
with open({remote_path!r}, "rb") as f:
    f.seek({offset})
    raw = f.read({slice_bytes})
print("SLICE_BEGIN")
print(base64.b64encode(raw).decode("ascii"))
print("SLICE_END")
"""
            s_out, _, _ = await execute_code(read_code, kernel_id=kernel_id, timeout_sec=120)
            b_tag = s_out.find("SLICE_BEGIN")
            e_tag = s_out.find("SLICE_END", b_tag) if b_tag != -1 else -1
            if b_tag == -1 or e_tag == -1:
                raise RuntimeError(f"Failed slice {idx} (offset={offset}) for {remote_path}")
            payload = base64.b64decode(s_out[b_tag + len("SLICE_BEGIN"):e_tag].strip())
            out_f.write(payload)
            h_local.update(payload)
            offset += len(payload)
            idx += 1

    if h_local.hexdigest() != expected_sha:
        raise RuntimeError(f"SHA256 mismatch for {local_path}: {h_local.hexdigest()} != {expected_sha}")
    print(f">>> Pulled {os.path.basename(local_path)} ({total_size/(1024*1024):.2f} MB, sha256={expected_sha[:12]}...) -> {local_path}")
    return True


def push_remote_bytes(data: bytes, remote_rel_path: str):
    """Uploads binary data cleanly into /kaggle/working/<remote_rel_path> via Jupyter Contents API."""
    import urllib.parse
    import urllib.request

    url = _require_proxy_url() + "/api/contents/" + urllib.parse.quote(remote_rel_path)
    payload = json.dumps(
        {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(data).decode("ascii"),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Jupyter PUT failed for {remote_rel_path}: HTTP {resp.status}")
    print(f">>> Uploaded {remote_rel_path} ({len(data)/(1024*1024):.2f} MB via HTTP /api/contents)", flush=True)


async def main():
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        tar.add(PYTHON_DIR, arcname="python/bz_ggml")
        for test_f in ("register_cilian_voice.py", "verify_1x15min_fp16_cilian.py", "texts_corpus_5x20m.py"):
            tar.add(os.path.join(ROOT_DIR, "tests", test_f), arcname=f"tests/{test_f}")

    push_remote_bytes(tar_buf.getvalue(), "bz_py_bundle.tar.gz")
    with open(CILIAN_MP3_PATH, "rb") as f:
        push_remote_bytes(f.read(), "cilian_9sec.mp3")

    remote_script = '''
import os, sys, subprocess, tarfile, time

print("=" * 80, flush=True)
print(">>> STEP 1: Unpacking synced bz_ggml Python & linking prebuilt CUDA sm_75 binary...", flush=True)
print("=" * 80, flush=True)

os.makedirs("/kaggle/working/bz-ggml/tests", exist_ok=True)
os.makedirs("/kaggle/working/models", exist_ok=True)

with tarfile.open("/kaggle/working/bz_py_bundle.tar.gz", mode="r:gz") as tar:
    tar.extractall(path="/kaggle/working/bz-ggml")

prebuilt_so = "/kaggle/working/breeze-cuda-sm75/lib/libbreeze.so"
if not os.path.exists(prebuilt_so):
    bin_tar = "/kaggle/working/breeze-cuda-sm75.tar.gz"
    if not os.path.exists(bin_tar):
        bin_url = "https://github.com/sharjeel103/bz-ggml/raw/prebuilt-t4/breeze-cuda-sm75.tar.gz"
        print(f"Downloading prebuilt binaries from {bin_url}...", flush=True)
        subprocess.run(f"wget -q -c {bin_url} -O {bin_tar}", shell=True, check=True)
    subprocess.run(f"tar -xzf {bin_tar} -C /kaggle/working/", shell=True, check=True)

subprocess.run("rm -rf /kaggle/working/bz-ggml/build && mkdir -p /kaggle/working/bz-ggml/build", shell=True, check=True)
subprocess.run("cp -P /kaggle/working/breeze-cuda-sm75/lib/* /kaggle/working/bz-ggml/build/", shell=True, check=True)
subprocess.run("rm -f /usr/lib/x86_64-linux-gnu/libbreeze*.so* /usr/lib/x86_64-linux-gnu/libggml*.so*", shell=True)
subprocess.run("ln -sf /kaggle/working/bz-ggml/build/*.so* /usr/lib/x86_64-linux-gnu/", shell=True, check=True)
subprocess.run("ldconfig", shell=True)
print("Prebuilt libbreeze.so linked cleanly (0s compile):", os.path.exists("/kaggle/working/bz-ggml/build/libbreeze.so"), flush=True)

print("\\n>>> STEP 2: Verifying / installing faster-whisper, resemble-enhance, and av<18...", flush=True)
try:
    import faster_whisper
    from resemble_enhance.enhancer.inference import load_enhancer, inference
    print("Deps already installed & verified OK", flush=True)
except ImportError:
    subprocess.run(
        "pip install -q 'av<18' faster-whisper soundfile deepspeed hjson tabulate celluloid resampy omegaconf ptflops librosa && "
        "pip install -q --no-deps resemble-enhance",
        shell=True,
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "-c",
            "import faster_whisper; from resemble_enhance.enhancer.inference import load_enhancer, inference; print('Deps verified OK')",
        ],
        check=True,
    )

print("\\n>>> STEP 3: Downloading breeze-tts-2-f16.gguf (~5.99 GB)...", flush=True)
f16_path = "/kaggle/working/models/breeze-tts-2-f16.gguf"
if not os.path.exists(f16_path) or os.path.getsize(f16_path) < 1000000000:
    t_dl = time.time()
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="HoppouAI/Breeze-TTS-2.cpp", filename="breeze-tts-2-f16.gguf", local_dir="/kaggle/working/models")
    print(f"FP16 model downloaded in {time.time() - t_dl:.1f}s!", flush=True)

def run_sub(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command {cmd} failed with exit code {p.returncode}")

print("\\n" + "=" * 80, flush=True)
print(">>> STEP 3b: Cleaning up any stale GPU compute processes & checking VRAM...", flush=True)
print("=" * 80, flush=True)
import signal
try:
    apps_out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in apps_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == os.getpid():
            continue
        cmd_str = subprocess.check_output(["ps", "-p", str(pid), "-o", "args="], text=True).strip()
        if "python" in cmd_str or "breeze" in cmd_str:
            print(f"    Killing stale GPU PID {pid} ({parts[1]} MiB): {cmd_str}", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(1.5)
except Exception as exc:
    print(f"    [VRAM cleanup note] {exc}", flush=True)

vram_out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"],
    text=True,
).strip()
print(f"    GPU VRAM state:\\n{vram_out}", flush=True)
for line in vram_out.splitlines():
    idx_s, free_s, used_s, tot_s = [x.strip() for x in line.split(",")]
    if int(free_s) < 14000:
        raise RuntimeError(f"GPU {idx_s} only has {free_s} MiB free (< 14000 MiB required); aborting.")

print("\\n" + "=" * 80, flush=True)
print(">>> STEP 4: Running Phase 1 Voice Registration Subprocess (clean exit reclaims GPU VRAM)...", flush=True)
print("=" * 80, flush=True)
run_sub(["python3", "-u", "/kaggle/working/bz-ggml/tests/register_cilian_voice.py"])

print("\\n" + "=" * 80, flush=True)
print(">>> STEP 5: Running Phase 2 Full 15-Min Cloned FP16 Benchmark...", flush=True)
print("=" * 80, flush=True)
run_sub(["python3", "-u", "/kaggle/working/bz-ggml/tests/verify_1x15min_fp16_cilian.py"])
'''

    kernel_id = get_or_create_idle_kernel()
    purged = purge_extra_kernels(keep_kernel_id=kernel_id)
    if purged:
        print(f">>> Purged stale Kaggle kernels releasing orphaned VRAM: {purged}")
    restart_kernel(kernel_id)
    await asyncio.sleep(2.0)
    print("=" * 80)
    print(f">>> Connected to clean Kaggle Kernel: {kernel_id}")
    print(">>> Running Cilian Voice Registration + Full 15m Cloned FP16 Pipeline...")
    print("=" * 80)

    t0 = time.time()
    stdout_out, stderr_out, status = await execute_code(remote_script, kernel_id=kernel_id, timeout_sec=3600)
    print(f"\n>>> Remote execution ended with status '{status}' in {time.time() - t0:.1f}s.")
    if status != "ok":
        raise RuntimeError("Remote execution failed; inspect logs above.")

    # Pull all artifacts locally using reliable chunked retrieval
    remote_files = [
        ("/kaggle/working/benchmark_results/cilian.breeze", os.path.join(LOCAL_RESULTS_DIR, "cilian.breeze")),
        ("/kaggle/working/benchmark_results/cilian_registration_meta.json", os.path.join(LOCAL_RESULTS_DIR, "cilian_registration_meta.json")),
        ("/kaggle/working/benchmark_results/summary_1x15m_fp16_cilian.json", os.path.join(LOCAL_RESULTS_DIR, "summary_1x15m_fp16_cilian.json")),
        ("/kaggle/working/benchmark_results/telemetry_1x15m_fp16_cilian.csv", os.path.join(LOCAL_RESULTS_DIR, "telemetry_1x15m_fp16_cilian.csv")),
        ("/kaggle/working/benchmark_results/benchmark_1x15m_fp16_cilian_dashboard.png", os.path.join(LOCAL_RESULTS_DIR, "benchmark_1x15m_fp16_cilian_dashboard.png")),
        ("/kaggle/working/benchmark_results/audio/req_15m_cilian_fp16_raw.flac", os.path.join(LOCAL_AUDIO_DIR, "req_15m_cilian_fp16_raw.flac")),
        ("/kaggle/working/benchmark_results/audio/req_15m_cilian_fp16_mastered.flac", os.path.join(LOCAL_AUDIO_DIR, "req_15m_cilian_fp16_mastered.flac")),
    ]
    for r_p, l_p in remote_files:
        ok = await pull_remote_file(r_p, l_p, kernel_id)
        if ok and l_p.endswith(".flac"):
            wav_p = l_p[:-5] + ".wav"
            data, sr = sf.read(l_p, dtype="float32")
            sf.write(wav_p, data, sr, subtype="PCM_16")
            print(f"    Decoded lossless FLAC -> {wav_p} ({os.path.getsize(wav_p)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
