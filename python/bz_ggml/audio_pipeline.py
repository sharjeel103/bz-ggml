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
from typing import Optional, List, Dict, Any, Generator, Tuple, Callable
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
            # Protect abbreviations with temporary placeholder
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

def apply_resemble_enhancer(
    input_wav_path: str,
    output_wav_path: str,
    chunk_sec: float = 15.0,
    device: Optional[str] = None
) -> str:
    """
    Enhances audio, removes noise, and upsamples to 44.1 kHz using Resemble Enhance.
    Uses 15s chunking and explicit memory cleanup to prevent GPU VRAM exhaustion.
    """
    try:
        import torch
        from resemble_enhance.enhancer.inference import enhance
    except ImportError:
        print("[Enhancer Warning] resemble-enhance is not installed. Copying raw audio...")
        shutil.copyfile(input_wav_path, output_wav_path)
        return output_wav_path

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        else:
            device = "cpu"

    print(f"[Resemble Enhancer] Processing {os.path.basename(input_wav_path)} on {device} (Chunks: {chunk_sec}s)...")
    t0 = time.time()

    try:
        data, sr = sf.read(input_wav_path, dtype="float32")
        dwav = torch.from_numpy(data)
        if dwav.dim() > 1:
            dwav = dwav.mean(dim=-1)

        if "cuda" in device:
            torch.cuda.empty_cache()

        hwav, out_sr = enhance(
            dwav=dwav,
            sr=sr,
            device=device,
            nfe=16,
            solver="midpoint",
            lambd=0.5,
            tau=0.5,
            chunk_seconds=chunk_sec,
            overlap_seconds=1.0,
        )

        out_np = hwav.cpu().numpy()
        max_val = np.abs(out_np).max()
        if max_val > 1.0:
            out_np = out_np / max_val * 0.99

        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
        sf.write(output_wav_path, out_np, out_sr, subtype="PCM_16")
        print(f"[Resemble Enhancer] Completed in {time.time()-t0:.2f}s -> {output_wav_path}")
        return output_wav_path
    finally:
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


# =====================================================================
# 4. GENTLE STUDIO MASTERING & NORMALIZATION (SWIFT_HISTORY PRESET)
# =====================================================================

def apply_studio_normalization(
    input_wav: str,
    output_wav: str,
    preset: str = "optionA_rich_bass"
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
        "-y",
        str(output_wav)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FFmpeg Warning] Studio normalization failed: {res.stderr}. Copying unmastered file.")
        shutil.copyfile(input_wav, output_wav)
    return output_wav


# =====================================================================
# 5. MP3 256K ENCODER
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
    a clean 1-word-per-line SRT subtitle file for Shorts, Reels, and Explainers.
    """
    res = asr_service.transcribe(
        audio=audio_path,
        vad_filter=True,
        word_timestamps=True,
        return_1word_srt=True
    )
    transcript, srt_content, word_list = res

    os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return output_srt_path


# =====================================================================
# 7. TYPED JOB QUEUES & PIPELINE DISPATCHER
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
    Orchestrates the entire multi-stage long-form production pipeline using
    strictly decoupled Job Queues with independent concurrency and VRAM bounds:
      - Queue 1 (TTS): Continuous batching across GPU 0 & GPU 1 via DualInstanceCluster.
      - Queue 2 (Stitch): CPU in-memory array ops with 200ms breath gap.
      - Queue 3 (Enhance): Resemble Enhance with semaphore(1) and memory cleanup.
      - Queue 4 (Master): FFmpeg broadcast mastering on CPU.
      - Queue 5 (ASR): Faster-Whisper with semaphore(2) on GPU 1.
    """
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.active_jobs: Dict[str, PipelineJob] = {}
        self.lock = threading.Lock()

        # Concurrency Guards
        self.enhance_semaphore = threading.BoundedSemaphore(1)
        self.master_semaphore = threading.BoundedSemaphore(4)

    def get_saved_files(self) -> List[str]:
        """Returns sorted list of saved deliverable files (.mp3, .srt, .wav)."""
        if not os.path.exists(self.output_dir):
            return []
        files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.endswith(".mp3") or f.endswith(".srt") or f.endswith(".wav")
        ]
        files.sort(key=os.path.getmtime, reverse=True)
        return files

    def run_pipeline(
        self,
        job: PipelineJob,
        cluster,
        asr_service,
        ref_text: Optional[str] = None,
        ref_codes: Optional[List[int]] = None,
        ref_frames: int = 0,
        update_callback: Optional[Callable[[PipelineJob], None]] = None
    ):
        """Executes the full pipeline sequentially through typed job stages."""
        from bz_ggml import UserTask

        def notify(stage: JobStage, progress: float, msg: str):
            job.stage = stage
            job.progress = progress
            job.status_log = msg
            print(f"[{job.job_id}] [{stage.value.upper()}] {msg}")
            if update_callback:
                update_callback(job)

        with self.lock:
            self.active_jobs[job.job_id] = job

        job_dir = os.path.join(self.output_dir, job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        try:
            # ---------------- STAGE 1: SEMANTIC CHUNKING ----------------
            notify(JobStage.TTS_CHUNKING, 0.05, f"Intelligently chunking script into semantic blocks (Target: {job.target_words}w, Max: {job.max_words}w)...")
            chunks = split_text_into_smart_chunks(job.script_text, target_words=job.target_words, max_words=job.max_words)
            if not chunks:
                raise ValueError("Script text contains no valid words.")

            job.chunks_count = len(chunks)
            words_total = sum(len(c.split()) for c in chunks)
            notify(JobStage.TTS_CHUNKING, 0.10, f"Script partitioned into {len(chunks)} cohesive chunks ({words_total} words total).")

            # ---------------- STAGE 2: PARALLEL DUAL-GPU TTS ----------------
            notify(JobStage.TTS_PARALLEL, 0.15, f"Dispatching {len(chunks)} parallel TTS jobs to DualInstanceCluster (GPU 0 & GPU 1)...")

            tts_tasks = []
            for idx, chunk_text in enumerate(chunks):
                t = UserTask(
                    id=idx,
                    text=chunk_text,
                    instruction=job.instruction,
                    ref_text=ref_text,
                    ref_codes=ref_codes,
                    ref_frames=ref_frames,
                    cfg_scale=job.cfg_scale,
                    temperature=job.temperature,
                    seed=job.seed + idx,
                    max_steps=2040,  # Full 2,040 token context headroom to guarantee zero cutoffs
                    stream_pcm=False,
                    client_has_vocoder=False
                )
                tts_tasks.append(t)

            chunks_completed = 0
            def on_chunk_done(res):
                nonlocal chunks_completed
                chunks_completed += 1
                job.chunks_completed = chunks_completed
                pct = 0.15 + (chunks_completed / len(chunks)) * 0.40
                notify(JobStage.TTS_PARALLEL, pct, f"TTS Synthesis: Completed chunk {chunks_completed}/{len(chunks)} ({res.worker}).")

            tts_results = cluster.run_workload(
                tts_tasks,
                out_dir=os.path.join(job_dir, "chunks"),
                on_progress=on_chunk_done
            )

            # Sort strictly by chronological chunk index
            tts_results_sorted = sorted(tts_results, key=lambda r: r.id)

            # Load PCM buffers for all completed chunks
            chunks_pcm = []
            total_dur = 0.0
            for r in tts_results_sorted:
                data, sr = sf.read(r.wav_path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                chunks_pcm.append(data)
                total_dur += r.audio_s

            job.total_audio_s = round(total_dur, 2)

            # ---------------- STAGE 3: STITCHING WITH 200ms BREATH GAP ----------------
            notify(JobStage.STITCHING, 0.60, f"Stitching {len(chunks_pcm)} chunks with {job.pause_ms}ms breath pause & click-free micro-fades...")
            stitched_pcm = stitch_audio_chunks(chunks_pcm, sample_rate=24000, pause_ms=job.pause_ms)

            raw_wav = os.path.join(job_dir, f"{job.job_id}_raw.wav")
            raw_mp3 = os.path.join(self.output_dir, f"{job.job_id}_raw.mp3")
            sf.write(raw_wav, stitched_pcm, 24000, subtype="PCM_16")
            convert_to_mp3_256k(raw_wav, raw_mp3)

            job.raw_wav_path = raw_wav
            job.raw_mp3_path = raw_mp3
            notify(JobStage.STITCHING, 0.65, f"Raw voiceover stitched ({len(stitched_pcm)/24000:.1f}s audio) -> Saved: {os.path.basename(raw_mp3)}")

            # ---------------- STAGE 4: RESEMBLE ENHANCE (OPTIONAL) ----------------
            current_master_wav = raw_wav
            if job.apply_enhance:
                notify(JobStage.ENHANCING, 0.70, "Acquiring Enhancer queue lock & running Resemble Enhance (15s sliding windows, 44.1 kHz)...")
                with self.enhance_semaphore:
                    enhanced_wav = os.path.join(job_dir, f"{job.job_id}_enhanced_tmp.wav")
                    apply_resemble_enhancer(raw_wav, enhanced_wav, chunk_sec=15.0)
                    current_master_wav = enhanced_wav
                notify(JobStage.ENHANCING, 0.80, "Resemble Enhance completed successfully.")
            else:
                notify(JobStage.ENHANCING, 0.80, "Resemble Enhance bypassed by user.")

            # ---------------- STAGE 5: GENTLE STUDIO MASTERING (OPTIONAL) ----------------
            final_enhanced_mp3 = os.path.join(self.output_dir, f"{job.job_id}_mastered.mp3")
            if job.apply_master:
                notify(JobStage.MASTERING, 0.82, "Applying studio broadcast mastering (70Hz Highpass + EQ 135Hz + Loudnorm -16 LUFS)...")
                with self.master_semaphore:
                    mastered_wav = os.path.join(job_dir, f"{job.job_id}_mastered.wav")
                    apply_studio_normalization(current_master_wav, mastered_wav, preset="optionA_rich_bass")
                    convert_to_mp3_256k(mastered_wav, final_enhanced_mp3)
                    job.enhanced_wav_path = mastered_wav
                    job.enhanced_mp3_path = final_enhanced_mp3
                notify(JobStage.MASTERING, 0.88, f"Studio mastering completed -> Saved: {os.path.basename(final_enhanced_mp3)}")
            else:
                if current_master_wav != raw_wav:
                    convert_to_mp3_256k(current_master_wav, final_enhanced_mp3)
                    job.enhanced_wav_path = current_master_wav
                    job.enhanced_mp3_path = final_enhanced_mp3
                else:
                    job.enhanced_wav_path = raw_wav
                    job.enhanced_mp3_path = raw_mp3

            # ---------------- STAGE 6: 1-WORD PER LINE SRT SUBTITLES (OPTIONAL) ----------------
            if job.apply_srt and asr_service is not None:
                notify(JobStage.TRANSCRIBING, 0.90, "Transcribing mastered audio with Whisper on GPU 1 (generating 1-word-per-line SRT)...")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc; gc.collect()

                    srt_path = os.path.join(self.output_dir, f"{job.job_id}_transcript_1word.srt")
                    target_audio_for_srt = job.enhanced_wav_path if job.enhanced_wav_path else raw_wav
                    generate_1word_srt_file(target_audio_for_srt, srt_path, asr_service=asr_service)
                    job.srt_path = srt_path
                    notify(JobStage.TRANSCRIBING, 0.98, f"1-Word per line SRT generated -> Saved: {os.path.basename(srt_path)}")
                except Exception as asr_err:
                    print(f"[{job.job_id}] [ASR Warning] Subtitle generation failed: {asr_err}")
                    job.srt_path = None
                    notify(JobStage.TRANSCRIBING, 0.98, f"Subtitle note: {asr_err} (Audio generation succeeded)")
            else:
                notify(JobStage.TRANSCRIBING, 0.98, "Subtitles bypassed by user.")

            notify(JobStage.COMPLETED, 1.0, f"Production Pipeline Complete! Generated all deliverables ({job.total_audio_s}s audio).")
            return job

        except Exception as e:
            job.stage = JobStage.FAILED
            job.error = str(e)
            notify(JobStage.FAILED, 1.0, f"Pipeline Error: {str(e)}")
            raise e
