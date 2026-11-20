import gc
import os
import queue
import threading
import time
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

class ASRWorkerPool:
    """
    Strict concurrency limiter for ASR transcription micro-service on GPU 0.
    Prevents concurrent transcription jobs from causing GPU VRAM spikes.
    """
    def __init__(self, max_concurrent: int = 2):
        self.semaphore = threading.BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self.lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        acquired = self.semaphore.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            with self.lock:
                self.active_count += 1
        return acquired

    def release(self):
        with self.lock:
            self.active_count = max(0, self.active_count - 1)
        self.semaphore.release()


def load_audio_16k(audio_input: Union[str, np.ndarray, List[float]], sample_rate: int = 24000) -> np.ndarray:
    """
    Loads an audio file or array and converts to 16 kHz mono float32 in [-1.0, 1.0].
    Directly bypasses PyAV / av.open to prevent 'metadata_errors' version incompatibility.
    """
    if isinstance(audio_input, np.ndarray):
        data = audio_input.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sample_rate != 16000 and len(data) > 0:
            import scipy.signal
            num_samples = int(round(len(data) * 16000.0 / float(sample_rate)))
            data = scipy.signal.resample(data, num_samples).astype(np.float32)
        return data

    if isinstance(audio_input, list):
        data = np.array(audio_input, dtype=np.float32)
        if sample_rate != 16000 and len(data) > 0:
            import scipy.signal
            num_samples = int(round(len(data) * 16000.0 / float(sample_rate)))
            data = scipy.signal.resample(data, num_samples).astype(np.float32)
        return data

    audio_path = str(audio_input)
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000 and len(data) > 0:
            import scipy.signal
            num_samples = int(round(len(data) * 16000.0 / float(sr)))
            data = scipy.signal.resample(data, num_samples).astype(np.float32)
        return data
    except Exception:
        pass

    try:
        import torchaudio
        sig, sr = torchaudio.load(audio_path)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        return sig.squeeze().cpu().numpy().astype(np.float32)
    except Exception:
        pass

    import wave
    with wave.open(audio_path, "rb") as wf:
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if n_ch > 1:
            data = data.reshape(-1, n_ch).mean(axis=1)
        if sr != 16000 and len(data) > 0:
            import scipy.signal
            num_samples = int(round(len(data) * 16000.0 / float(sr)))
            data = scipy.signal.resample(data, num_samples).astype(np.float32)
        return data


def format_srt_time(seconds: float) -> str:
    """
    Formats seconds into standard SRT timestamp HH:MM:SS,mmm.
    Uses pure modulo integer arithmetic to guarantee zero rollover or overflow bugs.
    """
    total_ms = max(0, int(round(seconds * 1000.0)))
    millis = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


class ASRService:
    """
    Dedicated Speech-to-Text Micro-Service using faster-whisper.
    Pinned strictly to GPU 0 (or CPU fallback) with bounded concurrency.
    Default Model: 'turbo' (large-v3-turbo, ~2,199 MiB resident VRAM in float16).
    Produces word-level SRT timestamps, plain text, or structured sentence segments.
    """
    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cuda",
        device_index: int = 0,
        cuda_device: Optional[int] = None,
        compute_type: str = "float16",
        max_concurrent: int = 2
    ):
        if cuda_device is not None:
            device_index = cuda_device
        self.model_size = model_size
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.pool = ASRWorkerPool(max_concurrent=max_concurrent)
        self._model = None
        self._init_lock = threading.Lock()

    def _ensure_model_loaded(self):
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return
            import torch
            if not torch.cuda.is_available() and self.device == "cuda":
                print(f"[ASR Service] CUDA unavailable; falling back to CPU.")
                self.device = "cpu"
                self.compute_type = "int8"

            from faster_whisper import WhisperModel
            print(f"[ASR Service] Initializing {self.model_size} ({self.compute_type}) on {self.device}:{self.device_index}...")
            t0 = time.time()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type
            )
            print(f"   -> ASR Model Online in {time.time()-t0:.2f}s (~2,199 MiB resident VRAM)")

    def transcribe(
        self,
        audio: Union[str, np.ndarray, List[float]],
        sample_rate: int = 24000,
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        return_subtitles: bool = True,
        return_1word_srt: bool = True,
        output_format: Optional[str] = None,
        timeout: Optional[float] = 60.0
    ) -> Union[str, Tuple[str, str, List[dict]], List[dict]]:
        """
        Transcribes speech audio into verified text with optional timestamped subtitles.
        Acquires semaphore before execution to strictly guarantee bounded concurrency on GPU 0.
        Bypasses PyAV by pre-decoding input into 16 kHz float32 numpy arrays.
        """
        self._ensure_model_loaded()

        acquired = self.pool.acquire(blocking=True, timeout=timeout)
        if not acquired:
            raise TimeoutError(f"ASR concurrency limit ({self.pool.max_concurrent}) exceeded on {self.device}:{self.device_index}")

        t0 = time.time()
        try:
            audio_16k = load_audio_16k(audio, sample_rate=sample_rate)

            segments, info = self._model.transcribe(
                audio_16k,
                beam_size=beam_size,
                language=language,
                temperature=0.0,
                vad_filter=vad_filter,
                word_timestamps=True
            )

            seg_list = []
            text_segments = []
            srt_lines = []
            word_idx = 1

            for s_idx, s in enumerate(segments, start=1):
                clean_text = s.text.strip()
                if clean_text:
                    text_segments.append(clean_text)

                if hasattr(s, "words") and s.words:
                    for w in s.words:
                        w_text = w.word.strip()
                        if not w_text:
                            continue
                        seg_list.append({
                            "id": word_idx,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "text": w_text
                        })
                        srt_lines.append(f"{word_idx}\n{format_srt_time(w.start)} --> {format_srt_time(w.end)}\n{w_text}\n")
                        word_idx += 1
                else:
                    seg_list.append({
                        "id": s_idx,
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "text": clean_text
                    })
                    srt_lines.append(f"{s_idx}\n{format_srt_time(s.start)} --> {format_srt_time(s.end)}\n{clean_text}\n")

            transcript = " ".join(text_segments).strip()
            srt_content = "\n".join(srt_lines).strip()

            dur = info.duration if hasattr(info, "duration") else 0.0
            print(f"[ASR Service] Transcribed {dur:.1f}s audio in {(time.time()-t0)*1000:.1f}ms: \"{transcript[:60]}...\"")

            if output_format == "srt":
                return srt_content
            elif output_format == "txt":
                return transcript
            elif output_format == "segments":
                return seg_list
            elif output_format == "all" or return_subtitles or return_1word_srt:
                return transcript, srt_content, seg_list
            return transcript
        finally:
            self.pool.release()
