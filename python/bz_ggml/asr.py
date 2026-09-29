import os
import threading
import queue
import time
from typing import Optional, Union, List
import numpy as np

class ASRWorkerPool:
    """
    Strict concurrency limiter for ASR transcription micro-service on GPU 1.
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


class ASRService:
    """
    Dedicated Speech-to-Text Micro-Service using faster-whisper.
    Pinned strictly to GPU 1 (or CPU fallback) with bounded concurrency.
    Memory Footprint: ~288 MiB weights + ~106 MiB scratchpad (small int8_float16).
    """
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        device_index: int = 1,
        cuda_device: Optional[int] = None,
        compute_type: str = "int8_float16",
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
            from faster_whisper import WhisperModel
            print(f"[ASR Service] Initializing {self.model_size} ({self.compute_type}) on {self.device}:{self.device_index}...")
            t0 = time.time()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type
            )
            print(f"   -> ASR Model Online in {time.time()-t0:.2f}s (~288 MiB resident VRAM)")

    def transcribe(
        self,
        audio: Union[str, np.ndarray, List[float]],
        language: str = "en",
        beam_size: int = 1,
        timeout: Optional[float] = 15.0
    ) -> str:
        """
        Transcribes speech audio into verified text.
        Acquires semaphore before execution to strictly guarantee max 2 concurrent jobs on GPU 1.
        """
        self._ensure_model_loaded()

        acquired = self.pool.acquire(blocking=True, timeout=timeout)
        if not acquired:
            raise TimeoutError(f"ASR concurrency limit ({self.pool.max_concurrent}) exceeded on {self.device}:{self.device_index}")

        t0 = time.time()
        try:
            # Handle float list / numpy
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)

            segments, info = self._model.transcribe(
                audio,
                beam_size=beam_size,
                language=language,
                temperature=0.0
            )
            text_segments = [s.text.strip() for s in segments]
            transcript = " ".join(text_segments).strip()
            print(f"[ASR Service] Transcribed in {(time.time()-t0)*1000:.1f}ms: \"{transcript}\"")
            return transcript
        finally:
            self.pool.release()
