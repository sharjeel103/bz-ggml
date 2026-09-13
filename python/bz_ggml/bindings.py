import ctypes
import os
import sys
from typing import Optional, List, Tuple

class BreezeLib:
    """Encapsulates ctypes C API bindings for libbreeze.so"""

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = self._resolve_lib_path(lib_path)
        self.lib = ctypes.CDLL(self.lib_path)
        self._bind_functions()

    def _resolve_lib_path(self, user_path: Optional[str]) -> str:
        candidates = []
        if user_path:
            candidates.append(user_path)
        if "BREEZE_LIB_PATH" in os.environ:
            candidates.append(os.environ["BREEZE_LIB_PATH"])

        # Relative to current python script or standard build locations
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        candidates.extend([
            os.path.join(repo_root, "build", "libbreeze.so"),
            os.path.join(repo_root, "build", "libbreeze.dylib"),
            os.path.join(repo_root, "build", "breeze.dll"),
            "/kaggle/working/Breeze-TTS-2.cpp/build/libbreeze.so",
            "/kaggle/working/bz-ggml/build/libbreeze.so",
            "libbreeze.so"
        ])

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Could not locate libbreeze.so in any of the following paths:\n" +
            "\n".join(candidates) +
            "\nPlease build the C++ project with `cmake -B build && cmake --build build` or set BREEZE_LIB_PATH."
        )

    def _bind_functions(self):
        # Generator API
        self.lib.breeze_generator_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.breeze_generator_init.restype = ctypes.c_void_p

        self.lib.breeze_generator_free.argtypes = [ctypes.c_void_p]
        self.lib.breeze_generator_free.restype = None

        self.lib.breeze_generator_prefill.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_prefill.restype = ctypes.c_int

        self.lib.breeze_generator_step_frame.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_step_frame.restype = ctypes.c_int

        # Vocoder API
        self.lib.breeze_vocoder_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.breeze_vocoder_init.restype = ctypes.c_void_p

        self.lib.breeze_vocoder_free.argtypes = [ctypes.c_void_p]
        self.lib.breeze_vocoder_free.restype = None

        self.lib.breeze_vocoder_stream_decode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.breeze_vocoder_stream_decode.restype = ctypes.c_int


class GeneratorHandle:
    def __init__(self, lib: BreezeLib, model_path: str, cuda_device: int = 0):
        self.lib = lib
        self.device = cuda_device
        self.handle = self.lib.lib.breeze_generator_init(model_path.encode("utf-8"), cuda_device)
        if not self.handle:
            raise RuntimeError(f"Failed to initialize Generator on CUDA device {cuda_device}")

    def prefill(self, text: str, instruction: str = "Speak clearly and naturally.", seed: int = 42) -> int:
        cb0_buf = ctypes.c_int()
        res = self.lib.lib.breeze_generator_prefill(
            self.handle, text.encode("utf-8"), instruction.encode("utf-8"),
            ctypes.c_uint32(seed), ctypes.byref(cb0_buf)
        )
        if res != 0:
            raise RuntimeError(f"Prefill failed on Generator (device {self.device})")
        return cb0_buf.value

    def step_frame(self, cb0: int, seed: int) -> Tuple[int, List[int]]:
        frame_buf = (ctypes.c_int * 16)()
        next_cb0 = self.lib.lib.breeze_generator_step_frame(
            self.handle, cb0, ctypes.c_uint32(seed), frame_buf
        )
        return next_cb0, list(frame_buf)

    def close(self):
        if self.handle:
            self.lib.lib.breeze_generator_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


class VocoderHandle:
    def __init__(self, lib: BreezeLib, model_path: str, cuda_device: int = 1):
        self.lib = lib
        self.device = cuda_device
        self.handle = self.lib.lib.breeze_vocoder_init(model_path.encode("utf-8"), cuda_device)
        if not self.handle:
            raise RuntimeError(f"Failed to initialize Streaming Vocoder on CUDA device {cuda_device}")

    def stream_decode(self, frames_tokens: List[int], n_frames: int) -> List[float]:
        pcm_buf = (ctypes.c_float * (n_frames * 1920))()
        c_frames = (ctypes.c_int * len(frames_tokens))(*frames_tokens)
        n_samples = self.lib.lib.breeze_vocoder_stream_decode(
            self.handle, c_frames, n_frames, pcm_buf
        )
        return list(pcm_buf)[:n_samples]

    def close(self):
        if self.handle:
            self.lib.lib.breeze_vocoder_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
