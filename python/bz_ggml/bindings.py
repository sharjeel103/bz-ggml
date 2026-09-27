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

        # Multi-Session Dynamic API
        self.lib.breeze_generator_session_create.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_float, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_session_create.restype = ctypes.c_int

        self.lib.breeze_generator_session_create_ext.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_float, ctypes.c_uint32, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_session_create_ext.restype = ctypes.c_int


        self.lib.breeze_generator_session_step.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_session_step.restype = ctypes.c_int

        self.lib.breeze_generator_session_free.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.breeze_generator_session_free.restype = ctypes.c_int

        self.lib.breeze_generator_load_q4_depth.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.breeze_generator_load_q4_depth.restype = ctypes.c_int

        self.lib.breeze_generator_session_set_q4.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.breeze_generator_session_set_q4.restype = ctypes.c_int

        self.lib.breeze_generator_session_count.argtypes = [ctypes.c_void_p]
        self.lib.breeze_generator_session_count.restype = ctypes.c_int

        self.lib.breeze_generator_sessions_step_round.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_sessions_step_round.restype = ctypes.c_int

        self.lib.breeze_generator_sessions_step_batched.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_sessions_step_batched.restype = ctypes.c_int

        self.lib.breeze_generator_sessions_step_burst.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_sessions_step_burst.restype = ctypes.c_int

        # Mode 4 Voice Conversion API
        self.lib.breeze_generator_convert_voice.argtypes = [
            ctypes.c_void_p,                                  # gen
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,       # src_codes, src_T
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,     # ref_audio, ref_audio_len
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,       # ref_codes, ref_frames
            ctypes.c_char_p,                                  # ref_text
            ctypes.c_char_p,                                  # src_text
            ctypes.c_float, ctypes.c_int, ctypes.c_int,       # cfg_scale, keep_acoustic, feed_source
            ctypes.c_uint32,                                  # seed
            ctypes.POINTER(ctypes.c_int),                     # out_codes
            ctypes.POINTER(ctypes.c_float), ctypes.c_int      # out_pcm, max_out_pcm
        ]
        self.lib.breeze_generator_convert_voice.restype = ctypes.c_int

        # Audio Encoder API
        self.lib.breeze_generator_encode_audio.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_generator_encode_audio.restype = ctypes.c_int

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

        self.lib.breeze_vocoder_stream_decode_batch.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.breeze_vocoder_stream_decode_batch.restype = ctypes.c_int


class GeneratorHandle:
    def __init__(self, lib: BreezeLib, model_path: str, cuda_device: int = 0):
        self.lib = lib
        self.device = cuda_device
        self.handle = self.lib.lib.breeze_generator_init(model_path.encode("utf-8"), cuda_device)
        if not self.handle:
            raise RuntimeError(f"Failed to initialize Generator on CUDA device {cuda_device}")

    def load_q4_depth(self, q4_model_path: str) -> bool:
        res = self.lib.lib.breeze_generator_load_q4_depth(
            self.handle, q4_model_path.encode("utf-8")
        )
        if res != 0:
            raise RuntimeError(f"Failed to load modular Q4 depth decoder on Generator (device {self.device})")
        return True

    def session_set_q4(self, session_id: int, use_q4: bool = True) -> bool:
        res = self.lib.lib.breeze_generator_session_set_q4(
            self.handle, session_id, 1 if use_q4 else 0
        )
        return res == 0

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

    def session_create(
        self, session_id: int, text: str, instruction: str = "Speak clearly and naturally.", 
        cfg_scale: float = 1.0, seed: int = 42, max_new_tokens: int = 0, use_q4: bool = False
    ) -> Tuple[int, int]:
        return self.session_create_ext(
            session_id=session_id, text=text, instruction=instruction,
            ref_text=None, ref_codes=None, ref_frames=0,
            cfg_scale=cfg_scale, seed=seed, max_new_tokens=max_new_tokens, use_q4=use_q4
        )

    def session_create_ext(
        self, session_id: int, text: str, instruction: str = "Speak clearly and naturally.", 
        ref_text: Optional[str] = None, ref_codes: Optional[List[int]] = None, ref_frames: int = 0,
        cfg_scale: float = 1.0, seed: int = 42, max_new_tokens: int = 0, use_q4: bool = False
    ) -> Tuple[int, int]:
        cb0_buf = ctypes.c_int()
        alloc_buf = ctypes.c_int()
        c_ref_text = ref_text.encode("utf-8") if (ref_text and len(ref_text) > 0) else None
        c_ins = instruction.encode("utf-8") if (instruction and len(instruction) > 0) else None
        if ref_codes and ref_frames > 0:
            c_ref_codes = (ctypes.c_int * len(ref_codes))(*ref_codes)
        else:
            c_ref_codes = None
        res = self.lib.lib.breeze_generator_session_create_ext(
            self.handle, session_id, text.encode("utf-8"), c_ins,
            c_ref_text, c_ref_codes, ref_frames,
            ctypes.c_float(cfg_scale), ctypes.c_uint32(seed), ctypes.c_int(max_new_tokens),
            ctypes.byref(cb0_buf), ctypes.byref(alloc_buf)
        )
        if res != 0:
            raise RuntimeError(f"Session create ext failed on Generator (device {self.device}, session {session_id})")
        if use_q4:
            self.session_set_q4(session_id, True)
        return cb0_buf.value, alloc_buf.value

    def session_step(self, session_id: int, seed: int) -> Tuple[int, List[int]]:
        frame_buf = (ctypes.c_int * 16)()
        next_cb0 = self.lib.lib.breeze_generator_session_step(
            self.handle, session_id, ctypes.c_uint32(seed), frame_buf
        )
        return next_cb0, list(frame_buf)

    def session_free(self, session_id: int):
        self.lib.lib.breeze_generator_session_free(self.handle, session_id)

    def active_session_count(self) -> int:
        return self.lib.lib.breeze_generator_session_count(self.handle)

    def sessions_step_round(self, session_ids: List[int], seed: int = 42) -> Tuple[List[int], List[List[int]]]:
        n = len(session_ids)
        if n == 0:
            return [], []
        c_sids = (ctypes.c_int * n)(*session_ids)
        c_frames = (ctypes.c_int * (n * 16))()
        c_next_cb0 = (ctypes.c_int * n)()
        self.lib.lib.breeze_generator_sessions_step_round(
            self.handle, c_sids, n, ctypes.c_uint32(seed), c_frames, c_next_cb0
        )
        next_cb0s = list(c_next_cb0)
        frames = [list(c_frames[i*16:(i+1)*16]) for i in range(n)]
        return next_cb0s, frames

    def sessions_step_batched(self, session_ids: List[int], seeds: Optional[List[int]] = None) -> Tuple[List[int], List[List[int]]]:
        n = len(session_ids)
        if n == 0:
            return [], []
        c_sids = (ctypes.c_int * n)(*session_ids)
        c_frames = (ctypes.c_int * (n * 16))()
        c_next_cb0 = (ctypes.c_int * n)()
        if seeds is not None and len(seeds) == n:
            c_seeds = (ctypes.c_uint32 * n)(*seeds)
        else:
            c_seeds = None
        self.lib.lib.breeze_generator_sessions_step_batched(
            self.handle, c_sids, n, c_seeds, c_frames, c_next_cb0
        )
        next_cb0s = list(c_next_cb0)
        frames = [list(c_frames[i*16:(i+1)*16]) for i in range(n)]
        return next_cb0s, frames

    def sessions_step_burst(
        self, session_ids: List[int], burst_steps: int, seeds: Optional[List[int]] = None
    ) -> Tuple[List[int], List[List[List[int]]]]:
        n = len(session_ids)
        if n == 0 or burst_steps <= 0:
            return [], []
        c_sids = (ctypes.c_int * n)(*session_ids)
        total_output_ints = burst_steps * n * 16
        c_frames = (ctypes.c_int * total_output_ints)()
        c_next_cb0 = (ctypes.c_int * n)()
        if seeds is not None and len(seeds) == n:
            c_seeds = (ctypes.c_uint32 * n)(*seeds)
        else:
            c_seeds = None
        
        self.lib.lib.breeze_generator_sessions_step_burst(
            self.handle, c_sids, n, burst_steps, c_seeds, c_frames, c_next_cb0
        )
        next_cb0s = list(c_next_cb0)
        # Reshape: for each session i in range(n): list of burst_steps frames (each 16 ints)
        session_frames = []
        for i in range(n):
            s_frames = []
            for step in range(burst_steps):
                offset = (step * n + i) * 16
                s_frames.append(list(c_frames[offset : offset + 16]))
            session_frames.append(s_frames)
        return next_cb0s, session_frames

    def convert_voice(
        self,
        src_codes: List[int],
        src_T: int,
        ref_audio: Optional[List[float]] = None,
        ref_codes: Optional[List[int]] = None,
        ref_frames: int = 0,
        ref_text: Optional[str] = None,
        src_text: Optional[str] = None,
        cfg_scale: float = 1.5,
        keep_acoustic: int = 0,
        feed_source: bool = True,
        seed: int = 42,
        return_pcm: bool = True
    ) -> Tuple[Optional[List[float]], Optional[List[int]]]:
        """
        Executes Mode 4 Speech-to-Speech Voice Conversion directly on the Generator's
        existing VRAM instance (via shallow base_model wrapper) with zero weight reloading.
        """
        if not self.handle:
            raise RuntimeError("GeneratorHandle is closed")

        c_src_codes = (ctypes.c_int * len(src_codes))(*src_codes)

        if ref_audio and len(ref_audio) > 0:
            c_ref_audio = (ctypes.c_float * len(ref_audio))(*ref_audio)
            c_ref_audio_len = len(ref_audio)
        else:
            c_ref_audio = None
            c_ref_audio_len = 0

        if ref_codes and ref_frames > 0:
            c_ref_codes = (ctypes.c_int * len(ref_codes))(*ref_codes)
            c_ref_frames = ref_frames
        else:
            c_ref_codes = None
            c_ref_frames = 0

        c_ref_text = ref_text.encode("utf-8") if (ref_text and len(ref_text) > 0) else None
        c_src_text = src_text.encode("utf-8") if (src_text and len(src_text) > 0) else None

        c_out_codes = (ctypes.c_int * (src_T * 16))()

        if return_pcm:
            max_pcm_len = src_T * 1920 + 32000
            c_out_pcm = (ctypes.c_float * max_pcm_len)()
            max_pcm = max_pcm_len
        else:
            c_out_pcm = None
            max_pcm = 0

        res = self.lib.lib.breeze_generator_convert_voice(
            self.handle,
            c_src_codes, src_T,
            c_ref_audio, c_ref_audio_len,
            c_ref_codes, c_ref_frames,
            c_ref_text, c_src_text,
            ctypes.c_float(cfg_scale),
            ctypes.c_int(keep_acoustic),
            ctypes.c_int(1 if feed_source else 0),
            ctypes.c_uint32(seed),
            c_out_codes,
            c_out_pcm, max_pcm
        )

        if res < 0:
            err = self.lib.get_last_error()
            raise RuntimeError(f"Voice conversion failed on Generator (device {self.device}): {err}")

        out_codes = list(c_out_codes)
        out_audio = list(c_out_pcm)[:res] if (return_pcm and c_out_pcm) else None
        return out_audio, out_codes

    def encode_audio(self, pcm_samples: List[float]) -> Tuple[List[int], int]:
        n_samples = len(pcm_samples)
        if n_samples == 0:
            return [], 0
        max_frames = int(math.ceil(n_samples / 1920.0)) + 64
        c_pcm = (ctypes.c_float * n_samples)(*pcm_samples)
        c_out_codes = (ctypes.c_int * (max_frames * 16))()
        c_out_n_frames = ctypes.c_int(0)

        res = self.lib.lib.breeze_generator_encode_audio(
            self.handle,
            c_pcm, n_samples,
            c_out_codes, ctypes.byref(c_out_n_frames)
        )
        if res < 0:
            err = self.lib.get_last_error()
            raise RuntimeError(f"Audio encoding failed on Generator (device {self.device}): {err}")

        n_frames = c_out_n_frames.value
        total_tokens = n_frames * 16
        return list(c_out_codes[:total_tokens]), n_frames

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
        max_chunk = 64
        if n_frames > max_chunk:
            all_samples = []
            for start in range(0, n_frames, max_chunk):
                end = min(n_frames, start + max_chunk)
                sub_n = end - start
                sub_toks = frames_tokens[start * 16 : end * 16]
                all_samples.extend(self.stream_decode(sub_toks, sub_n))
            return all_samples

        pcm_buf = (ctypes.c_float * (n_frames * 1920))()
        c_frames = (ctypes.c_int * len(frames_tokens))(*frames_tokens)
        n_samples = self.lib.lib.breeze_vocoder_stream_decode(
            self.handle, c_frames, n_frames, pcm_buf
        )
        return list(pcm_buf)[:n_samples]

    def stream_decode_batch(self, batch_tokens: List[List[int]], batch_n_frames: List[int]) -> List[List[float]]:
        B = len(batch_tokens)
        if B == 0:
            return []
        if B == 1:
            return [self.stream_decode(batch_tokens[0], batch_n_frames[0])]

        flat_tokens = []
        tok_offsets = []
        pcm_offsets = []
        total_samples = 0

        for b in range(B):
            tok_offsets.append(len(flat_tokens))
            flat_tokens.extend(batch_tokens[b])
            pcm_offsets.append(total_samples)
            total_samples += batch_n_frames[b] * 1920

        c_flat_tokens = (ctypes.c_int * len(flat_tokens))(*flat_tokens)
        c_tok_offsets = (ctypes.c_int * B)(*tok_offsets)
        c_n_frames = (ctypes.c_int * B)(*batch_n_frames)
        c_flat_pcm = (ctypes.c_float * total_samples)()
        c_pcm_offsets = (ctypes.c_int * B)(*pcm_offsets)
        c_out_samples = (ctypes.c_int * B)()

        ok = self.lib.lib.breeze_vocoder_stream_decode_batch(
            self.handle, B, c_flat_tokens, c_tok_offsets, c_n_frames,
            c_flat_pcm, c_pcm_offsets, c_out_samples
        )
        if ok <= 0:
            raise RuntimeError(f"breeze_vocoder_stream_decode_batch failed for batch of {B} streams")

        results = []
        pcm_arr = list(c_flat_pcm)
        for b in range(B):
            start = pcm_offsets[b]
            cnt = c_out_samples[b]
            results.append(pcm_arr[start:start + cnt])
        return results

    def close(self):
        if self.handle:
            self.lib.lib.breeze_vocoder_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def load_breeze_voice(path: str) -> dict:
    """Loads a pre-encoded .breeze voice file into memory for instant zero-shot cloning."""
    import struct
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"BRZV":
            raise ValueError(f"Invalid voice file magic: {magic} in {path}")
        version, sample_rate, n_codebooks, frames, text_len = struct.unpack("<5I", f.read(20))
        text = f.read(text_len).decode("utf-8")
        codes = list(struct.unpack(f"<{frames * n_codebooks}i", f.read()))
        return {
            "name": os.path.basename(path).replace(".breeze", ""),
            "text": text,
            "codes": codes,
            "frames": frames,
            "sample_rate": sample_rate,
            "n_codebooks": n_codebooks
        }

