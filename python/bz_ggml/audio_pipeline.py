import gc
import io
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
from .asr import ASRService


# =====================================================================
# 1. INTELLIGENT SEMANTIC TEXT CHUNKER
# =====================================================================

# Common English abbreviations that should not trigger sentence splits
_ABBREVIATIONS = (
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\."
)


def _split_oversized_clause(
    text: str,
    max_words_limit: int = 280,
    target_words_limit: int = 200
) -> List[str]:
    """
    Emergency fallback for runaway unpunctuated text (e.g. raw ASR/STT transcripts):
    Tier 1: Splits at natural coordinating conjunctions (and, but, or, because, so, while, since).
    Tier 2: Clean hard word slicing on space boundary if no conjunctions exist.
    Guarantees no unit exceeds max_words_limit, preventing GPU KV cache step-truncation.
    """
    words = text.split()
    if len(words) <= max_words_limit:
        return [text]

    # Tier 1: Split at natural coordinating conjunctions
    parts = re.split(r'\s+(?=(?:and|but|or|because|so|while|since)\b)', text, flags=re.IGNORECASE)
    result = []
    curr = []
    curr_len = 0

    for part in parts:
        p_len = len(part.split())
        if p_len > max_words_limit:
            if curr:
                result.append(" ".join(curr).strip())
                curr, curr_len = [], 0
            # Tier 2: Hard word slice
            sub_words = part.split()
            for j in range(0, len(sub_words), target_words_limit):
                result.append(" ".join(sub_words[j:j + target_words_limit]))
        elif curr_len + p_len > max_words_limit and curr:
            result.append(" ".join(curr).strip())
            curr = [part]
            curr_len = p_len
        else:
            curr.append(part)
            curr_len += p_len

    if curr:
        result.append(" ".join(curr).strip())
    return result


def split_text_into_smart_chunks(
    text: str,
    target_words: int = 200,
    max_words: int = 280,
    min_words: Optional[int] = None
) -> List[str]:
    """
    Elastic Semantic Chunking with Balanced Lookahead Partitioning:
      - Never cuts below 280 words if text fits within GPU context (zero chunking overhead).
      - For text > 280 words, calculates optimal balanced chunk count K = ceil(total_words / target_words).
      - Prioritizes paragraph breaks (\\n\\n+) over sentence ends, keeping thoughts intact.
      - Protects common abbreviations (Dr., Mr., etc.), decimals (3.14), and numbered lists (1. Item).
      - Handles quotes and closing punctuation without orphan fragments.
      - Emergency conjunction and word-level fallback for runaway unpunctuated text.
      - Guarantees balanced, cohesive speech chunks with natural cadence.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    words = cleaned_text.split()
    total_words = len(words)
    if total_words <= max_words:
        return [cleaned_text]

    if min_words is None:
        min_words = min(80, max(1, int(target_words * 0.4)))
    else:
        min_words = min(min_words, max_words)

    # Calculate optimal balanced chunk target
    k_chunks = max(2, math.ceil(total_words / float(target_words)))
    ideal_words = total_words / float(k_chunks)

    # 1. Break into raw paragraphs
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', cleaned_text) if p.strip()]

    # 2. Decompose oversized paragraphs into sentence / clause units with boundary tags
    units: List[Tuple[str, int, bool]] = []  # (text, word_count, is_paragraph_end)

    for p in raw_paragraphs:
        p_words = p.split()
        if len(p_words) <= max_words:
            units.append((p, len(p_words), True))
        else:
            # Paragraph is oversized: split into protected sentence units
            p_safe = p
            placeholders = {}

            # Protect abbreviations
            abbrev_matches = list(re.finditer(_ABBREVIATIONS, p_safe, re.IGNORECASE))
            for i, match in enumerate(reversed(abbrev_matches)):
                key = f"__ABBR_{i}__"
                placeholders[key] = match.group(0)
                p_safe = p_safe[:match.start()] + key + p_safe[match.end():]

            # Protect decimal numbers (e.g. 3.14)
            dec_matches = list(re.finditer(r'\b\d+\.\d+\b', p_safe))
            for i, match in enumerate(reversed(dec_matches)):
                key = f"__DEC_{i}__"
                placeholders[key] = match.group(0)
                p_safe = p_safe[:match.start()] + key + p_safe[match.end():]

            # Protect numbered lists (e.g. 1. Item)
            list_matches = list(re.finditer(r'(?:(?<=^)|(?<=\s))\d+\.', p_safe))
            for i, match in enumerate(reversed(list_matches)):
                key = f"__LIST_{i}__"
                placeholders[key] = match.group(0)
                p_safe = p_safe[:match.start()] + key + p_safe[match.end():]

            # Split by sentence-ending punctuation (handling quotes and closing parens)
            sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?]["\'\u201d\u2019\)])\s+', p_safe)
            restored_sentences = []
            for s in sentences:
                s_str = s.strip()
                if not s_str:
                    continue
                for key, original in placeholders.items():
                    s_str = s_str.replace(key, original)
                restored_sentences.append(s_str)

            for s_idx, s_str in enumerate(restored_sentences):
                is_p_end = (s_idx == len(restored_sentences) - 1)
                s_words = s_str.split()
                if len(s_words) <= max_words:
                    units.append((s_str, len(s_words), is_p_end))
                else:
                    # Single sentence exceeds max_words: split at natural clause pauses
                    clauses = re.split(r'(?<=[,;:\u2014])\s+', s_str)
                    curr_clause = []
                    curr_len = 0
                    for c_idx, c in enumerate(clauses):
                        # Safeguard: if a clause still exceeds max_words, split via conjunctions/words
                        c_sub_pieces = _split_oversized_clause(c, max_words_limit=max_words, target_words_limit=target_words)
                        for sp in c_sub_pieces:
                            sp_w = len(sp.split())
                            if curr_len + sp_w > max_words and curr_clause:
                                units.append((" ".join(curr_clause).strip(), curr_len, False))
                                curr_clause = [sp]
                                curr_len = sp_w
                            else:
                                curr_clause.append(sp)
                                curr_len += sp_w
                    if curr_clause:
                        units.append((" ".join(curr_clause).strip(), curr_len, is_p_end))

    # 3. Assemble atomic units using Elastic Lookahead Partitioning
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_words = 0

    total_units = len(units)
    for i, (unit_text, unit_w, is_p_end) in enumerate(units):
        remaining_units = units[i:]
        remaining_words = sum(u[1] for u in remaining_units)

        # Hard guard: if adding next unit exceeds max_words, cut now
        if current_words + unit_w > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [unit_text]
            current_words = unit_w
            continue

        current_chunk.append(unit_text)
        current_words += unit_w

        rem_after = remaining_words - unit_w

        # Elastic decision: cut at natural boundary if both current and remaining meet min_words
        if current_words >= min_words and rem_after >= min_words:
            # Paragraph boundary preference (highest priority)
            if is_p_end and current_words >= (ideal_words * 0.75):
                chunks.append("\n\n".join(current_chunk).strip())
                current_chunk = []
                current_words = 0
            # Sentence boundary when approaching ideal target
            elif current_words >= ideal_words:
                chunks.append("\n\n".join(current_chunk).strip())
                current_chunk = []
                current_words = 0

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks


def split_text_with_anchor(
    text: str,
    anchor_target_words: int = 35,
    anchor_max_words: int = 50,
    target_words: int = 110,
    max_words: int = 140
) -> List[str]:
    """
    Splits a long script for Anchor-First 1 + (N-1) parallel generation:
      - Chunk 0: short 25-50 word opening anchor (~5-8 seconds of speech) ending on a clean
        sentence/clause boundary so it establishes the exact live voice, tone, energy, and room
        capture without exceeding Breeze's safe reference-prompt length (~150-250 chars).
      - Chunks 1..N-1: balanced 100-140 word chunks generated in parallel across both GPUs.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    words = cleaned.split()
    if len(words) <= anchor_max_words:
        return [cleaned]

    # Split into sentence-level units reusing the abbreviation-safe splitter
    sentence_units = split_text_into_smart_chunks(cleaned, target_words=1, max_words=anchor_max_words)
    if len(sentence_units) <= 1:
        return split_text_into_smart_chunks(cleaned, target_words=target_words, max_words=max_words)

    anchor_parts = []
    anchor_wc = 0
    idx = 0
    while idx < len(sentence_units):
        u = sentence_units[idx]
        u_wc = len(u.split())
        if anchor_parts and (anchor_wc + u_wc > anchor_max_words):
            break
        anchor_parts.append(u)
        anchor_wc += u_wc
        idx += 1
        if anchor_wc >= anchor_target_words:
            break

    anchor_text = " ".join(anchor_parts).strip()
    remaining_text = "\n\n".join(sentence_units[idx:]).strip()
    if not remaining_text:
        return [anchor_text]

    follower_chunks = split_text_into_smart_chunks(
        remaining_text,
        target_words=target_words,
        max_words=max_words
    )
    return [anchor_text] + follower_chunks


def preview_chunks_info(
    text: str,
    target_words: int = 80,
    max_words: int = 120,
    min_words: Optional[int] = None
) -> Dict[str, Any]:
    """Provides structured inspection of chunk breakdown for UI preview."""
    chunks = split_text_into_smart_chunks(text, target_words=target_words, max_words=max_words, min_words=min_words)
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


def format_chunk_preview_markdown(
    text: str,
    target_words: int = 80,
    max_words: int = 120,
    min_words: Optional[int] = None
) -> str:
    """Formats chunk breakdown into clean Markdown for the Gradio UI."""
    info = preview_chunks_info(text, target_words, max_words, min_words)
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
# 2. CHUNK STITCHING, LEVELING & SIGNAL QUALITY DIAGNOSTICS
# =====================================================================

def estimate_audio_snr_db(pcm: np.ndarray, sr: int = 24000) -> float:
    """
    Estimates active-speech SNR in dB using 20ms frame percentiles (P90 vs P5)
    with a quiet-frame spectral-flatness guard for continuous non-paused speech.
    """
    if pcm is None or len(pcm) == 0:
        return 0.0
    arr = np.asarray(pcm, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)

    win = max(16, int(0.02 * sr))
    n_frames = len(arr) // win
    if n_frames < 4:
        return 0.0

    frames = arr[: n_frames * win].reshape(n_frames, win)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    frame_db = 20.0 * np.log10(frame_rms + 1e-9)

    p90 = float(np.percentile(frame_db, 90))
    p5 = float(np.percentile(frame_db, 5))
    if p90 <= -55.0:
        return 0.0

    # If the clip has clear quiet pauses, P90 - P5 directly captures speech-to-noise ratio.
    if (p90 - p5) >= 20.0:
        return round(max(0.0, p90 - p5), 2)

    # Continuous speech with few pauses: check upper-band Wiener spectral flatness on quietest frames.
    q_count = max(1, n_frames // 10)
    quiet_idx = np.argsort(frame_db)[:q_count]
    freqs = np.fft.rfftfreq(win, 1.0 / float(sr))
    hi_mask = freqs >= min(2000.0, float(sr) * 0.25)
    hann = np.hanning(win).astype(np.float32)
    flatnesses = []
    for idx in quiet_idx:
        pwr = np.abs(np.fft.rfft(frames[idx] * hann)) ** 2
        band = pwr[hi_mask] + 1e-12
        if len(band) > 0:
            geo = float(np.exp(np.mean(np.log(band))))
            arith = float(np.mean(band))
            flatnesses.append(geo / max(1e-12, arith))
    med_flat = float(np.median(flatnesses)) if flatnesses else 1.0
    if med_flat < 0.28 and float(np.percentile(frame_db, 5)) > -35.0:
        # Quietest segments are still clean tonal speech (no audible broadband noise floor)
        return 35.0
    return round(max(0.0, p90 - p5), 2)


def trim_chunk_edges(
    pcm: np.ndarray,
    sr: int = 24000,
    thresh_db: float = -52.0,
    pad_ms: float = 35.0,
) -> np.ndarray:
    """Safely trims dead digital silence on chunk edges while preserving 35ms natural attack/release."""
    if pcm is None or len(pcm) == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(pcm, dtype=np.float32)
    if len(arr) <= int(0.5 * sr):
        return arr

    thresh = 10.0 ** (thresh_db / 20.0)
    active = np.flatnonzero(np.abs(arr) >= thresh)
    if len(active) == 0:
        return arr

    pad = int(max(0.0, pad_ms) * sr / 1000.0)
    start = max(0, int(active[0]) - pad)
    end = min(len(arr), int(active[-1]) + pad + 1)
    if end - start < int(0.25 * sr):
        return arr
    return arr[start:end]


def apply_micro_fade(pcm: np.ndarray, fade_samples: int = 120) -> np.ndarray:
    """Applies a smooth raised-cosine fade-in and fade-out to prevent edge clicks."""
    if len(pcm) < fade_samples * 2 or fade_samples <= 0:
        return pcm
    pcm = pcm.copy()
    t = np.linspace(0.0, np.pi, fade_samples, dtype=np.float32)
    fade_in = 0.5 * (1.0 - np.cos(t))
    fade_out = 0.5 * (1.0 + np.cos(t))
    pcm[:fade_samples] *= fade_in
    pcm[-fade_samples:] *= fade_out
    return pcm


def level_and_stitch_chunks(
    chunks_pcm: List[np.ndarray],
    sample_rate: int = 24000,
    pause_ms: int = 200,
    max_gain_db: float = 2.5,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """
    Trims excess edge silence, gently equalizes chunk-to-chunk active loudness
    (clamped to +-max_gain_db around median chunk RMS, zero hard clipping),
    and joins chunks with raised-cosine fades and a natural breath pause.
    """
    valid = [trim_chunk_edges(np.asarray(c, dtype=np.float32), sr=sample_rate) for c in chunks_pcm if c is not None and len(c) > 0]
    if not valid:
        return np.array([], dtype=np.float32)

    if len(valid) > 1 and max_gain_db > 0.0:
        active_thresh = 10.0 ** (-45.0 / 20.0)
        chunk_dbs: List[float] = []
        for c in valid:
            speech = c[np.abs(c) >= active_thresh]
            ref = speech if len(speech) >= max(240, len(c) // 20) else c
            rms = float(np.sqrt(np.mean(ref * ref)))
            chunk_dbs.append(20.0 * math.log10(rms + 1e-9))

        target_db = float(np.clip(np.median(chunk_dbs), -24.0, -14.0))
        leveled: List[np.ndarray] = []
        for c, c_db in zip(valid, chunk_dbs):
            gain_db = float(np.clip(target_db - c_db, -max_gain_db, max_gain_db))
            scaled = c * float(10.0 ** (gain_db / 20.0))
            peak = float(np.max(np.abs(scaled)))
            if peak > 0.96:
                scaled = scaled * (0.95 / peak)
            leveled.append(scaled.astype(np.float32))
        valid = leveled

    fade_len = int(sample_rate * (max(0.0, fade_ms) / 1000.0))
    silence_len = int(sample_rate * (max(0, int(pause_ms)) / 1000.0))
    silence = np.zeros(silence_len, dtype=np.float32) if silence_len > 0 else None

    stitched_parts = []
    for idx, chunk in enumerate(valid):
        stitched_parts.append(apply_micro_fade(chunk.astype(np.float32), fade_samples=fade_len))
        if idx < len(valid) - 1 and silence is not None:
            stitched_parts.append(silence)

    return np.concatenate(stitched_parts)


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

_RESEMBLE_CPU_MODEL: Any = None
_RESEMBLE_LOCK = threading.Lock()

def get_resemble_cpu_model():
    """
    Stages Resemble Enhancer model weights in CPU system RAM (0 MiB GPU VRAM).
    Models are transferred to GPU on-demand during inference and evacuated immediately.
    """
    global _RESEMBLE_CPU_MODEL
    with _RESEMBLE_LOCK:
        if _RESEMBLE_CPU_MODEL is not None:
            return _RESEMBLE_CPU_MODEL
        try:
            import torch
            from resemble_enhance.enhancer.inference import load_enhancer
            print("[Resemble Enhancer] Initializing weights in host CPU RAM (0 MiB GPU VRAM)...")
            t0 = time.time()
            _RESEMBLE_CPU_MODEL = load_enhancer(None, device="cpu")
            print(f"   -> Resemble Enhancer ready in CPU RAM in {time.time()-t0:.2f}s!")
            return _RESEMBLE_CPU_MODEL
        except Exception as e:
            print(f"[Resemble Enhancer] Failed to load model on CPU: {e}")
            return None


def get_resemble_enhancer_model(device: str = "cuda:1"):
    return get_resemble_cpu_model()


def _prepare_resemble_chunking(
    orig_samples: int,
    sr: int,
    chunk_seconds: float,
    target_rate: int = 44100,
) -> Tuple[float, float, int]:
    """
    Computes safe (chunk_seconds, overlap_seconds, pad_input_samples) for
    resemble_enhance.inference() so trailing chunks never drop below 1.5s
    (avoiding [1, 128, 1] BatchNorm spatial crashes and overlap misalignment).
    """
    safe_chunk_s = max(2.0, float(chunk_seconds))
    dur_s = orig_samples / float(sr)
    min_chunk_s = 1.5

    if dur_s <= safe_chunk_s:
        # Single-pass mode: resemblance steps by (chunk - overlap), so overlap=0
        # guarantees <=10s inputs run as 1 whole chunk without a tiny tail sliver.
        pad_in = max(0, int(math.ceil(min_chunk_s * sr)) - orig_samples) if dur_s < min_chunk_s else 0
        eff_dur_s = max(dur_s, min_chunk_s)
        eff_chunk_s = max(safe_chunk_s, eff_dur_s + 0.05)
        return eff_chunk_s, 0.0, pad_in

    eff_overlap_s = min(1.0, safe_chunk_s * 0.25)
    chunk_len_t = int(target_rate * safe_chunk_s)
    overlap_len_t = int(target_rate * eff_overlap_s)
    hop_len_t = max(1, chunk_len_t - overlap_len_t)
    est_target_len = int(math.ceil(orig_samples * target_rate / float(sr)))
    rem_t = est_target_len % hop_len_t
    min_tail_t = max(int(min_chunk_s * target_rate), overlap_len_t + int(0.5 * target_rate))

    pad_in = 0
    if rem_t < min_tail_t:
        needed_t = (min_tail_t - rem_t) + int(0.05 * target_rate)
        pad_in = int(math.ceil(needed_t * float(sr) / target_rate))
    return safe_chunk_s, eff_overlap_s, pad_in


def enhance_audio_array(
    pcm: np.ndarray,
    sr: int = 24000,
    chunk_seconds: float = 10.0,
    device: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Enhances in-memory float32 audio, removes noise, and upsamples to 44.1 kHz.
    Moves model from CPU to GPU on-demand and evacuates 100% of GPU memory immediately after.
    """
    if pcm is None or len(pcm) == 0:
        return pcm, sr

    try:
        import torch
        import torch.nn.functional as F
        from resemble_enhance.enhancer.inference import inference
    except ImportError:
        print("[Enhancer Warning] resemble-enhance is not installed. Returning raw audio...")
        return pcm, sr

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        else:
            device = "cpu"

    device_idx = int(device.split(":")[-1]) if ":" in device else 0

    cpu_model = get_resemble_cpu_model()
    if cpu_model is None:
        return pcm, sr

    t0 = time.time()
    cuda_model = None
    try:
        # 1. Transfer model weights from CPU to GPU on-demand (~0.4s)
        cuda_model = cpu_model.to(device)
        cuda_model.eval()

        dwav = torch.from_numpy(pcm.astype(np.float32))
        if dwav.dim() > 1:
            dwav = dwav.mean(dim=-1)

        orig_samples = int(dwav.shape[-1])
        target_rate = int(getattr(getattr(cuda_model, "hp", None), "wav_rate", 44100))
        eff_chunk_s, eff_overlap_s, pad_in = _prepare_resemble_chunking(
            orig_samples=orig_samples,
            sr=sr,
            chunk_seconds=chunk_seconds,
            target_rate=target_rate,
        )
        if pad_in > 0:
            dwav = F.pad(dwav, (0, pad_in))

        # 2. Run chunked neural inference in inference_mode
        with torch.inference_mode():
            hwav, out_sr = inference(
                model=cuda_model,
                dwav=dwav,
                sr=sr,
                device=device,
                chunk_seconds=eff_chunk_s,
                overlap_seconds=eff_overlap_s,
            )

        if pad_in > 0:
            expected_out_len = int(round(orig_samples * float(out_sr) / float(sr)))
            hwav = hwav[:expected_out_len]

        out_np = hwav.cpu().numpy()
        max_val = np.abs(out_np).max()
        if max_val > 1.0:
            out_np = out_np / max_val * 0.99

        print(f"[Resemble Enhancer] In-memory enhanced {orig_samples/sr:.1f}s audio in {time.time()-t0:.2f}s (Out SR: {out_sr})")
        return out_np.astype(np.float32), int(out_sr)
    finally:
        # 3. Immediately evacuate model weights back to CPU and purge CUDA cache
        try:
            if cuda_model is not None:
                cuda_model.to("cpu")
                del cuda_model
        except Exception:
            pass
        gc.collect()
        if "cuda" in device and torch.cuda.is_available():
            try:
                with torch.cuda.device(device_idx):
                    torch.cuda.empty_cache()
            except Exception:
                torch.cuda.empty_cache()


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
    Applies broadcast-grade studio mastering purely in-memory via FFmpeg pipe:0 -> pipe:1
    (zero disk I/O):
      - 70 Hz highpass filter (removes sub-bass rumble)
      - EBU R128 gentle loudnorm (-16 LUFS broadcast target, -3 dB TruePeak)
      - 135 Hz parametric bass warmth boost (+3.5 dB)
      - True peak limiter ceiling (0.85) to guarantee 0 clipping.
    """
    if pcm is None or len(pcm) == 0:
        return np.zeros(0, dtype=np.float32), int(sr)

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

    pcm_f32 = np.ascontiguousarray(pcm, dtype=np.float32)
    cmd = [
        FFMPEG_EXE,
        "-v", "error",
        "-f", "f32le",
        "-ar", str(int(sr)),
        "-ac", "1",
        "-i", "pipe:0",
        "-af", filter_str,
        "-f", "f32le",
        "-ar", str(int(sr)),
        "-ac", "1",
        "pipe:1",
    ]
    res = subprocess.run(cmd, input=pcm_f32.tobytes(), capture_output=True)
    if res.returncode != 0 or not res.stdout:
        print(f"[FFmpeg Warning] In-memory studio mastering failed: {res.stderr.decode(errors='ignore').strip()}. Returning unmastered audio.")
        return pcm_f32, int(sr)

    out_pcm = np.frombuffer(res.stdout, dtype=np.float32).copy()
    return out_pcm, int(sr)


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


def encode_pcm_to_mp3_bytes(pcm: np.ndarray, sr: int, bitrate: str = "256k") -> bytes:
    """
    Encodes float32 PCM array directly to MP3 bytes in memory via FFmpeg pipe.
    Zero disk writes, sub-10ms latency.
    """
    pcm_i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    cmd = [
        FFMPEG_EXE,
        "-v", "error",
        "-f", "s16le",
        "-ar", str(sr),
        "-ac", "1",
        "-i", "pipe:0",
        "-acodec", "libmp3lame",
        "-b:a", str(bitrate),
        "-f", "mp3",
        "pipe:1"
    ]
    res = subprocess.run(cmd, input=pcm_i16.tobytes(), capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"In-memory FFmpeg MP3 encoding failed: {res.stderr.decode()}")
    return res.stdout


def encode_pcm_to_wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    """
    Encodes float32 PCM array directly to 16-bit PCM WAV bytes in memory.
    Zero disk writes.
    """
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(data) == 0:
        return data.astype(np.float32)
    import scipy.signal
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // gcd
    down = int(orig_sr) // gcd
    return scipy.signal.resample_poly(data, up, down).astype(np.float32)


def load_any_audio(
    audio_input: Union[str, bytes, np.ndarray, List[float]],
    target_sr: Optional[int] = None,
    max_duration_s: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Universal Audio Ingestion Engine.
    Accepts: file path (str), raw file bytes (bytes), numpy array (np.ndarray), or float list.
    Decodes via 3-tier waterfall: soundfile -> torchaudio -> ffmpeg pipe.
    Guarantees:
      - Mono float32 [-1.0, 1.0] output.
      - Exact target_sr if requested (using rational polyphase resampling).
      - Strict max_duration_s clipping at ingress (ensuring all pipeline steps process identical slice).
    """
    if isinstance(audio_input, np.ndarray):
        data = audio_input.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1 if data.shape[1] < data.shape[0] else 0)
        sr = target_sr or 24000
    elif isinstance(audio_input, list):
        data = np.array(audio_input, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1 if data.shape[1] < data.shape[0] else 0)
        sr = target_sr or 24000
    else:
        # If input is a remote presigned or public URL (S3 / GCS / R2), download bytes into memory
        if isinstance(audio_input, str) and (audio_input.startswith("http://") or audio_input.startswith("https://")):
            import urllib.request
            with urllib.request.urlopen(audio_input, timeout=30.0) as resp:
                audio_input = resp.read()

        loaded = False
        data, sr = None, None

        # Tier 1: soundfile (fastest for WAV / FLAC / uncompressed)
        try:
            target = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else str(audio_input)
            data, sr = sf.read(target, dtype="float32")
            loaded = True
        except Exception:
            pass

        # Tier 2: torchaudio (handles common MP3 / OGG)
        if not loaded:
            try:
                import torchaudio
                target = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else str(audio_input)
                sig, sr = torchaudio.load(target)
                if sig.shape[0] > 1:
                    sig = sig.mean(dim=0)
                data = sig.squeeze().cpu().numpy().astype(np.float32)
                loaded = True
            except Exception:
                pass

        # Tier 3: ffmpeg pipe (universal decoder for MP3, M4A, AAC, WEBM, OGG, OPUS, etc.)
        if not loaded:
            cmd = [FFMPEG_EXE, "-v", "error"]
            if isinstance(audio_input, bytes):
                cmd.extend(["-i", "pipe:0"])
                input_bytes = audio_input
            else:
                cmd.extend(["-i", str(audio_input)])
                input_bytes = None
            out_rate = target_sr or 24000
            cmd.extend(["-f", "f32le", "-ac", "1", "-ar", str(out_rate), "pipe:1"])
            res = subprocess.run(cmd, input=input_bytes, capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(f"Universal audio decoder failed: {res.stderr.decode()}")
            data = np.frombuffer(res.stdout, dtype=np.float32).copy()
            sr = out_rate

        if data.ndim > 1:
            data = data.mean(axis=1 if data.shape[1] < data.shape[0] else 0)

    # Resample if target_sr specified
    if target_sr is not None and sr != target_sr:
        data = _resample_audio(data, sr, target_sr)
        sr = target_sr

    # Hard-clip to max_duration_s if specified
    if max_duration_s is not None and max_duration_s > 0:
        max_samples = int(round(max_duration_s * sr))
        if len(data) > max_samples:
            data = data[:max_samples]

    return data.astype(np.float32), int(sr)


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
