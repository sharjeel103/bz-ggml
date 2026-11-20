import gc
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Tuple, Callable, Union
import numpy as np
import soundfile as sf

FFMPEG_EXE = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg"


# =====================================================================
# 1. INTELLIGENT SEMANTIC TEXT CHUNKER
# =====================================================================

# Common English abbreviations that should not trigger sentence splits
_ABBREVIATIONS = (
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\."
)

def split_text_into_smart_chunks(
    text: str,
    target_words: int = 80,
    max_words: int = 120
) -> List[str]:
    """
    Intelligently splits arbitrarily large text into cohesive speech chunks:
      - Hierarchical splitting: Paragraphs (\\n\\n+) -> Sentences ([.!?]) -> Clauses ([,;:]).
      - Never cuts a word or cohesive sentence in half.
      - Protects common abbreviations and numbers from accidental splits.
      - Default 80-120 words per chunk guarantees zero context overflow or truncation.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # 1. Break into paragraph blocks
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', cleaned_text) if p.strip()]

    # 2. Decompose oversized paragraphs into sentence units
    atomic_units = []
    for p in raw_paragraphs:
        words = p.split()
        if len(words) <= max_words:
            # Paragraph is already within safe size; keep it together
            atomic_units.append(p)
        else:
            # Paragraph is oversized: split into sentence units protecting abbreviations
            p_safe = p
            abbrev_matches = list(re.finditer(_ABBREVIATIONS, p_safe, re.IGNORECASE))
            placeholders = {}
            for i, match in enumerate(reversed(abbrev_matches)):
                key = f"__ABBR_{i}__"
                placeholders[key] = match.group(0)
                p_safe = p_safe[:match.start()] + key + p_safe[match.end():]

            # Protect decimal numbers (e.g., 3.14, 1.5)
            dec_matches = list(re.finditer(r'\b\d+\.\d+\b', p_safe))
            for i, match in enumerate(reversed(dec_matches)):
                key = f"__DEC_{i}__"
                placeholders[key] = match.group(0)
                p_safe = p_safe[:match.start()] + key + p_safe[match.end():]

            # Split by sentence-ending punctuation followed by whitespace
            sentences = re.split(r'(?<=[.!?])\s+', p_safe)

            # Restore placeholders
            restored_sentences = []
            for s in sentences:
                s_str = s.strip()
                if not s_str:
                    continue
                for key, original in placeholders.items():
                    s_str = s_str.replace(key, original)
                restored_sentences.append(s_str)

            for s_str in restored_sentences:
                s_words = s_str.split()
                if len(s_words) <= max_words:
                    atomic_units.append(s_str)
                else:
                    # Single sentence exceeds max_words: split at natural clause pauses
                    clauses = re.split(r'(?<=[,;:\u2014])\s+', s_str)
                    curr_clause = []
                    curr_len = 0
                    for c in clauses:
                        c_w = len(c.split())
                        if curr_len + c_w > max_words and curr_clause:
                            atomic_units.append(" ".join(curr_clause).strip())
                            curr_clause = [c]
                            curr_len = c_w
                        else:
                            curr_clause.append(c)
                            curr_len += c_w
                    if curr_clause:
                        atomic_units.append(" ".join(curr_clause).strip())

    # 3. Greedily aggregate atomic units into balanced chunks
    chunks = []
    current_chunk = []
    current_word_count = 0

    for unit in atomic_units:
        unit_words = len(unit.split())
        if current_word_count + unit_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [unit]
            current_word_count = unit_words
        elif current_word_count + unit_words >= target_words:
            current_chunk.append(unit)
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = []
            current_word_count = 0
        else:
            current_chunk.append(unit)
            current_word_count += unit_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks


def preview_chunks_info(text: str, target_words: int = 80, max_words: int = 120) -> Dict[str, Any]:
    """Provides structured inspection of chunk breakdown for UI preview."""
    chunks = split_text_into_smart_chunks(text, target_words=target_words, max_words=max_words)
    total_words = len(text.strip().split()) if text.strip() else 0
    return {
        "total_words": total_words,
        "chunks_count": len(chunks),
        "target_words": target_words,
        "max_words": max_words,
        "chunks": [
            {
                "index": i + 1,
                "words": len(c.split()),
                "text": c
            }
            for i, c in enumerate(chunks)
        ]
    }


def format_chunk_preview_markdown(text: str, target_words: int = 80, max_words: int = 120) -> str:
    """Formats chunk breakdown into clean Markdown for the Gradio UI."""
    info = preview_chunks_info(text, target_words, max_words)
    if info["chunks_count"] == 0:
        return "⚠️ *Script is empty. Paste your text above to see chunk distribution.*"

    md = [
        f"### 📊 Chunk Partition Summary",
        f"• **Total Words**: `{info['total_words']}` words",
        f"• **Parallel Chunks**: `{info['chunks_count']}` chunks (Target: {target_words}w, Max: {max_words}w)\n",
        "| Chunk | Word Count | Preview Text |",
        "| :---: | :---: | :--- |"
    ]
    for c in info["chunks"]:
        preview_snip = c["text"][:75].replace("\n", " ") + ("..." if len(c["text"]) > 75 else "")
        md.append(f"| **Chunk {c['index']}** | `{c['words']} words` | *\"{preview_snip}\"* |")

    return "\n".join(md)


# =====================================================================
# 2. CHUNK STITCHING & BREATH GAP
# =====================================================================

def apply_micro_fade(pcm: np.ndarray, fade_samples: int = 120) -> np.ndarray:
    """Applies a 5ms linear fade-in and fade-out to prevent DC offset clicks."""
    if len(pcm) < fade_samples * 2:
        return pcm
    pcm = pcm.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    pcm[:fade_samples] *= fade_in
    pcm[-fade_samples:] *= fade_out
    return pcm


def stitch_audio_chunks(
    chunks_pcm: List[np.ndarray],
    sample_rate: int = 24000,
    pause_ms: int = 200
) -> np.ndarray:
    """
    Concatenates sequential audio chunks with a natural breath pause between them.
    Adds a micro-fade to every chunk boundary to eliminate digital clicks/pops.
    """
    if not chunks_pcm:
        return np.array([], dtype=np.float32)

    fade_len = int(sample_rate * 0.005)  # 5 ms fade
    silence_len = int(sample_rate * (pause_ms / 1000.0))
    silence = np.zeros(silence_len, dtype=np.float32)

    stitched_parts = []
    for idx, chunk in enumerate(chunks_pcm):
        if len(chunk) == 0:
            continue
        smooth_chunk = apply_micro_fade(chunk.astype(np.float32), fade_samples=fade_len)
        stitched_parts.append(smooth_chunk)
        if idx < len(chunks_pcm) - 1 and silence_len > 0:
            stitched_parts.append(silence)

    if not stitched_parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(stitched_parts)


# =====================================================================
# 3. RESEMBLE ENHANCE (DENOISE & 44.1k UPSAMPLING)
# =====================================================================

_RESEMBLE_MODEL_CACHE: Dict[str, Any] = {}
_RESEMBLE_LOCK = threading.Lock()

def get_resemble_enhancer_model(device: str = "cuda:1"):
    """
    Singleton cache for Resemble Enhancer model weights in GPU VRAM.
    Keeps model resident permanently to eliminate reload overhead.
    """
    global _RESEMBLE_MODEL_CACHE
    with _RESEMBLE_LOCK:
        if device in _RESEMBLE_MODEL_CACHE:
            return _RESEMBLE_MODEL_CACHE[device]
        try:
            import torch
            from resemble_enhance.enhancer.inference import load_enhancer
            print(f"[Resemble Enhancer] Loading permanent model weights on {device} (~1,477 MiB VRAM)...")
            t0 = time.time()
            model = load_enhancer(None, device=device)
            _RESEMBLE_MODEL_CACHE[device] = model
            print(f"   -> Resemble Enhancer Online on {device} in {time.time()-t0:.2f}s!")
            return model
        except Exception as e:
            print(f"[Resemble Enhancer] Failed to load model on {device}: {e}")
            return None


def enhance_audio_array(
    pcm: np.ndarray,
    sr: int = 24000,
    chunk_seconds: float = 10.0,
    device: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Enhances in-memory float32 audio, removes noise, and upsamples to 44.1 kHz.
    Uses configurable chunk_seconds (default: 10.0s saves 1.2 GB active VRAM) and 1.0s overlap.
    """
    try:
        import torch
        from resemble_enhance.enhancer.inference import inference
    except ImportError:
        print("[Enhancer Warning] resemble-enhance is not installed. Returning raw audio...")
        return pcm, sr

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        else:
            device = "cpu"

    model = get_resemble_enhancer_model(device)
    if model is None:
        return pcm, sr

    t0 = time.time()
    try:
        dwav = torch.from_numpy(pcm.astype(np.float32))
        if dwav.dim() > 1:
            dwav = dwav.mean(dim=-1)

        if "cuda" in device:
            torch.cuda.empty_cache()

        hwav, out_sr = inference(
            model=model,
            dwav=dwav,
            sr=sr,
            device=device,
            chunk_seconds=chunk_seconds,
            overlap_seconds=1.0,
        )

        out_np = hwav.cpu().numpy()
        max_val = np.abs(out_np).max()
        if max_val > 1.0:
            out_np = out_np / max_val * 0.99

        print(f"[Resemble Enhancer] In-memory enhanced {len(pcm)/sr:.1f}s audio in {time.time()-t0:.2f}s (Out SR: {out_sr})")
        return out_np.astype(np.float32), int(out_sr)
    finally:
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def apply_resemble_enhancer(
    input_wav_path: str,
    output_wav_path: str,
    chunk_sec: float = 10.0,
    device: Optional[str] = None
) -> str:
    """
    Enhances audio file on disk via Resemble Enhance.
    Calls enhance_audio_array for model caching and memory safety.
    """
    data, sr = sf.read(input_wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    out_pcm, out_sr = enhance_audio_array(data, sr=sr, chunk_seconds=chunk_sec, device=device)
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, out_pcm, out_sr, subtype="PCM_16")
    return output_wav_path


# =====================================================================
# 4. GENTLE STUDIO MASTERING & NORMALIZATION (SWIFT_HISTORY PRESET)
# =====================================================================

def master_audio_array(
    pcm: np.ndarray,
    sr: int = 24000,
    preset: str = "optionA_rich_bass"
) -> Tuple[np.ndarray, int]:
    """
    Applies broadcast-grade studio mastering in-memory via FFmpeg:
      - 70 Hz highpass filter (removes sub-bass rumble)
      - EBU R128 gentle loudnorm (-16 LUFS broadcast target, -3 dB TruePeak)
      - 135 Hz parametric bass warmth boost (+3.5 dB)
      - True peak limiter ceiling (0.85) to guarantee 0 clipping.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_path = in_f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = out_f.name

    try:
        sf.write(in_path, pcm, sr, subtype="PCM_16")
        apply_studio_normalization(in_path, out_path, preset=preset, target_sr=sr)
        out_pcm, out_sr = sf.read(out_path, dtype="float32")
        if out_pcm.ndim > 1:
            out_pcm = out_pcm.mean(axis=1)
        return out_pcm.astype(np.float32), int(out_sr)
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def apply_studio_normalization(
    input_wav: str,
    output_wav: str,
    preset: str = "optionA_rich_bass",
    target_sr: Optional[int] = None
) -> str:
    """
    Applies broadcast-grade studio mastering from swift_history via FFmpeg:
      - 70 Hz highpass filter (removes sub-bass rumble)
      - EBU R128 gentle loudnorm (-16 LUFS broadcast target, -3 dB TruePeak)
      - 135 Hz parametric bass warmth boost (+3.5 dB)
      - True peak limiter ceiling (0.85) to guarantee 0 clipping.
    """
    if preset == "optionA_rich_bass":
        filter_str = (
            "highpass=f=70,"
            "loudnorm=I=-16.0:TP=-3.0:LRA=10,"
            "equalizer=f=135:t=q:w=0.8:g=3.5,"
            "alimiter=limit=0.85:level=false:asc=0"
        )
    elif preset == "gentle_loudnorm":
        filter_str = "highpass=f=70,loudnorm=I=-16.0:TP=-2.0:LRA=11,alimiter=limit=0.90:level=false:asc=0"
    else:
        filter_str = "loudnorm=I=-16.0:TP=-2.0:LRA=11"

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    cmd = [
        FFMPEG_EXE,
        "-v", "error",
        "-i", str(input_wav),
        "-af", filter_str,
        "-c:a", "pcm_s16le",
    ]
    if target_sr:
        cmd.extend(["-ar", str(target_sr)])
    cmd.extend(["-y", str(output_wav)])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FFmpeg Warning] Studio normalization failed: {res.stderr}. Copying unmastered file.")
        shutil.copyfile(input_wav, output_wav)
    return output_wav


# =====================================================================
# 5. MP3 & WAV ENCODERS
# =====================================================================

def convert_to_mp3_256k(wav_path: str, mp3_path: str) -> str:
    """Converts a WAV file into a high-bitrate 256 kbps MP3 for fast downloads."""
    os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
    cmd = [
        FFMPEG_EXE,
        "-v", "error",
        "-i", str(wav_path),
        "-acodec", "libmp3lame",
        "-b:a", "256k",
        "-y",
        str(mp3_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg MP3 encoding failed: {res.stderr.strip()}")
    return mp3_path


def save_pcm_to_mp3(pcm: np.ndarray, sr: int, mp3_path: str, bitrate: str = "256k") -> str:
    """Saves in-memory float32 PCM directly as high-bitrate MP3."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_file = tmp_wav.name
    try:
        sf.write(wav_file, pcm, sr, subtype="PCM_16")
        os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
        cmd = [
            FFMPEG_EXE,
            "-v", "error",
            "-i", str(wav_file),
            "-acodec", "libmp3lame",
            "-b:a", bitrate,
            "-y",
            str(mp3_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg MP3 encoding failed: {res.stderr.strip()}")
        return mp3_path
    finally:
        if os.path.exists(wav_file):
            try:
                os.remove(wav_file)
            except Exception:
                pass


def save_pcm_to_wav(pcm: np.ndarray, sr: int, wav_path: str) -> str:
    """Saves in-memory float32 PCM directly as standard PCM_16 WAV."""
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    sf.write(wav_path, pcm, sr, subtype="PCM_16")
    return wav_path


# =====================================================================
# 6. 1-WORD PER LINE SRT GENERATOR
# =====================================================================

def generate_1word_srt_file(
    audio_path: str,
    output_srt_path: str,
    asr_service
) -> str:
    """
    Generates word-level timestamps via Faster-Whisper and formats into
    a clean 1-word-per-line SRT subtitle file.
    """
    res = asr_service.transcribe(
        audio=audio_path,
        vad_filter=True,
        word_timestamps=True,
        return_1word_srt=True,
        output_format="srt"
    )
    if isinstance(res, tuple):
        _, srt_content, _ = res
    else:
        srt_content = str(res)

    os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return output_srt_path


# =====================================================================
# 7. TYPED JOB QUEUES & PIPELINE DISPATCHER (LEGACY COMPATIBILITY)
# =====================================================================

class JobStage(str, Enum):
    QUEUED = "queued"
    TTS_CHUNKING = "tts_chunking"
    TTS_PARALLEL = "tts_parallel"
    STITCHING = "stitching"
    ENHANCING = "enhancing"
    MASTERING = "mastering"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineJob:
    job_id: str
    script_text: str
    voice_name: Optional[str]
    instruction: str
    cfg_scale: float
    seed: int
    pause_ms: int = 200
    target_words: int = 80
    max_words: int = 120
    temperature: float = 0.25
    apply_enhance: bool = True
    apply_master: bool = True
    apply_srt: bool = True
    stage: JobStage = JobStage.QUEUED
    progress: float = 0.0
    status_log: str = "Initialized"
    chunks_count: int = 0
    chunks_completed: int = 0
    raw_wav_path: Optional[str] = None
    raw_mp3_path: Optional[str] = None
    enhanced_wav_path: Optional[str] = None
    enhanced_mp3_path: Optional[str] = None
    srt_path: Optional[str] = None
    total_audio_s: float = 0.0
    error: Optional[str] = None


class PipelineJobDispatcher:
    """
    Legacy dispatcher for Gradio backward compatibility.
    New code should dispatch Parcels to EngineOrchestrator.
    """
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.active_jobs: Dict[str, PipelineJob] = {}
        self.lock = threading.Lock()
        self.enhance_semaphore = threading.BoundedSemaphore(1)
        self.master_semaphore = threading.BoundedSemaphore(4)

    def get_saved_files(self) -> List[str]:
        if not os.path.exists(self.output_dir):
            return []
        files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.endswith(".mp3") or f.endswith(".srt") or f.endswith(".wav")
        ]
        files.sort(key=os.path.getmtime, reverse=True)
        return files
