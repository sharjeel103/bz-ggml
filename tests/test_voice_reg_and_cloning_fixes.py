"""
Offline unit tests verifying fixes for:
  1. Preserving explicit registered cloning reference (ref_codes/ref_frames/ref_text)
     across ALL chunks in parallel without Chunk 0 copy-of-a-copy overwrite.
  2. Un-cloned Anchor-First chunking defaulting anchor_words=20 (<= 25 max)
     so Chunk 0 fits under the 120-frame cap without audio/text truncation mismatch.
  3. Lossless float32 FFmpeg decode fallback in load_any_audio (zero s16le hard clamping)
     + MASTER (gentle_loudnorm) integration in Parcel.voice_registration.
  4. Single-pass Whisper ASR reuse (no double transcription in VOICE_REGISTER),
     skipping ASR when explicit transcript is provided, and syncing default EngineCapabilities.
"""

import os
import sys
import threading
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from bz_ggml.audio_pipeline import (
    _prepare_resemble_chunking,
    enhance_audio_array,
    estimate_audio_snr_db,
    level_and_stitch_chunks,
    load_any_audio,
    master_audio_array,
)
from bz_ggml.orchestrator import (
    EngineCapabilities,
    EngineOrchestrator,
    Parcel,
    RequestTTSAssembly,
    TaskSpec,
    TaskType,
)


class _DummyVocoderCluster:
    def __init__(self, precision="fp16", enable_asr=False, enable_enhancer=True, vocoder_mode="on"):
        self.precision = precision
        self.enable_q4_burst = False
        self.enable_asr = enable_asr
        self.enable_enhancer = enable_enhancer
        self.vocoder_mode = vocoder_mode
        self.sample_rate = 24000
        self.is_enhancer_running = False
        self.submitted_tasks = []
        self._lock = threading.Lock()
        self.encoded_calls = []

    def submit_tts_chunk(self, task, on_complete, cancel_event=None):
        with self._lock:
            self.submitted_tasks.append(task)
        # Return distinct synthetic tokens per chunk id so overwrite would be detectable
        fake_tokens = [task.id + 1] * (16 * 150)
        stats = {"session_id": task.id + 1, "total_frames": 150}
        threading.Thread(
            target=on_complete,
            args=(fake_tokens, None, stats),
            daemon=True,
        ).start()
        return task.id + 1

    def decode_tokens_tiled(self, tokens, ctx_frames=80, step_frames=480, gpu_id=None):
        n_frames = len(tokens) // 16
        return np.zeros(n_frames * 1920, dtype=np.float32)

    def encode_voice_container(self, pcm_samples, transcript, gpu_id=None):
        self.encoded_calls.append({"len": len(pcm_samples), "transcript": transcript})
        return b"BRZV_DUMMY_" + transcript.encode("utf-8")


class _MockASRService:
    def __init__(self, text="Clean transcribed reference speech."):
        self.text = text
        self.call_count = 0

    def transcribe(self, audio=None, sample_rate=24000, output_format="txt", **kwargs):
        self.call_count += 1
        if output_format == "all":
            return self.text, "1\n00:00:00,000 --> 00:00:01,000\nClean\n", []
        return self.text


class TestVoiceRegAndCloningFixes(unittest.TestCase):

    def test_capabilities_sync_from_cluster_and_asr(self):
        cluster_no_asr = _DummyVocoderCluster(precision="fp16", enable_asr=False, enable_enhancer=True)
        orch1 = EngineOrchestrator(cluster=cluster_no_asr, asr_service=None)
        try:
            self.assertEqual(orch1.capabilities.precision, "fp16")
            self.assertFalse(orch1.capabilities.enable_asr)
            self.assertTrue(orch1.capabilities.enable_enhancer)
        finally:
            orch1.shutdown()

        # Even if cluster enables ASR flag, if asr_service is None, capability stays False
        cluster_flag_only = _DummyVocoderCluster(precision="int8", enable_asr=True)
        orch2 = EngineOrchestrator(cluster=cluster_flag_only, asr_service=None)
        try:
            self.assertFalse(orch2.capabilities.enable_asr)
        finally:
            orch2.shutdown()

        cluster_full = _DummyVocoderCluster(precision="fp16", enable_asr=True)
        orch3 = EngineOrchestrator(cluster=cluster_full, asr_service=_MockASRService())
        try:
            self.assertTrue(orch3.capabilities.enable_asr)
            self.assertEqual(orch3.capabilities.precision, "fp16")
        finally:
            orch3.shutdown()

    def test_explicit_cloning_ref_preserved_across_all_chunks(self):
        cluster = _DummyVocoderCluster(precision="fp16", enable_asr=False, enable_enhancer=False)
        orch = EngineOrchestrator(
            cluster=cluster,
            capabilities=EngineCapabilities(precision="fp16", enable_enhancer=False, enable_asr=False),
        )
        try:
            ref_frames = 112
            ref_codes = [777] * (ref_frames * 16)
            ref_text = "Verified cilian reference transcript."
            long_script = ("This authentic sentence tests cloned voice consistency across parallel chunks. " * 38).strip()

            p = Parcel.tts_pipeline(
                script_text=long_script,
                voice_name="cilian",
                cfg_scale=2.5,
                follower_cfg_scale=2.0,
                enhance=False,
                master=False,
                srt=False,
                ref_codes=ref_codes,
                ref_frames=ref_frames,
                ref_text=ref_text,
            )
            res = orch.dispatch(p, timeout=15.0)
            self.assertTrue(res.success, res.error_message)
            self.assertGreater(len(cluster.submitted_tasks), 1)

            tasks_by_id = sorted(cluster.submitted_tasks, key=lambda t: t.id)
            for idx, task in enumerate(tasks_by_id):
                self.assertEqual(task.id, idx)
                self.assertEqual(task.ref_text, ref_text)
                self.assertEqual(task.ref_frames, ref_frames)
                self.assertEqual(task.ref_codes, ref_codes)
                self.assertEqual(task.seed, 42)
                self.assertLessEqual(len(task.text.split()), 60)
                expected_cfg = 2.5 if idx == 0 else 2.0
                self.assertEqual(task.cfg_scale, expected_cfg)
        finally:
            orch.shutdown()

    def test_uncloned_anchor_first_uses_short_anchor_and_copies_chunk0(self):
        cluster = _DummyVocoderCluster(precision="int8", enable_asr=False, enable_enhancer=False)
        orch = EngineOrchestrator(
            cluster=cluster,
            capabilities=EngineCapabilities(enable_enhancer=False, enable_asr=False),
        )
        try:
            long_script = ("First clean short sentence here. Second sentence builds the opening anchor. " * 25).strip()
            chunks = orch._prepare_tts_chunks(long_script)
            self.assertGreater(len(chunks), 1)
            self.assertLessEqual(len(chunks[0].split()), 25)

            p = Parcel.tts_pipeline(
                script_text=long_script,
                cfg_scale=2.0,
                follower_cfg_scale=1.5,
                enhance=False,
                master=False,
                srt=False,
            )
            res = orch.dispatch(p, timeout=15.0)
            self.assertTrue(res.success, res.error_message)

            tasks_by_id = sorted(cluster.submitted_tasks, key=lambda t: t.id)
            self.assertIsNone(tasks_by_id[0].ref_text)
            self.assertEqual(tasks_by_id[0].ref_frames, 0)
            self.assertEqual(tasks_by_id[0].cfg_scale, 2.0)
            # Mock returned 150 frames -> Chunk 0 cap trims follower anchor_frames to 120
            for follower in tasks_by_id[1:]:
                self.assertEqual(follower.ref_text, chunks[0])
                self.assertEqual(follower.ref_frames, 120)
                self.assertEqual(follower.cfg_scale, 1.5)
        finally:
            orch.shutdown()

    def test_voice_registration_single_asr_and_master_stage(self):
        # Verify task plan structure: ENHANCE -> MASTER(gentle_loudnorm) -> ASR (only if no transcript) -> VOICE_REGISTER
        p_auto = Parcel.voice_registration(voice_name="cilian", audio_input=np.zeros(24000, dtype=np.float32))
        self.assertEqual(
            [t.task_type for t in p_auto.tasks],
            [TaskType.ENHANCE, TaskType.MASTER, TaskType.ASR, TaskType.VOICE_REGISTER],
        )
        self.assertEqual(p_auto.tasks[1].params.get("preset"), "gentle_loudnorm")

        p_manual = Parcel.voice_registration(
            voice_name="cilian",
            audio_input=np.zeros(24000, dtype=np.float32),
            transcript="Manual verbatim text.",
            enhance=False,
        )
        self.assertEqual(
            [t.task_type for t in p_manual.tasks],
            [TaskType.MASTER, TaskType.VOICE_REGISTER],
        )

        # Dispatch with MockASRService and verify ASR is invoked ONLY ONCE (not twice!)
        cluster = _DummyVocoderCluster(enable_asr=True, enable_enhancer=False)
        asr = _MockASRService(text="Exact cilian speech.")
        orch = EngineOrchestrator(
            cluster=cluster,
            asr_service=asr,
            capabilities=EngineCapabilities(enable_enhancer=False, enable_asr=True),
        )
        try:
            # Generate slight overshoot > 1.0 sine wave to confirm MASTER cleans peak smoothly
            t = np.linspace(0.0, 1.0, 24000, endpoint=False, dtype=np.float32)
            sine = (1.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

            p_run = Parcel.voice_registration(
                voice_name="cilian",
                audio_input=sine,
                enhance=False,
                master=True,
            )
            res = orch.dispatch(p_run, timeout=15.0)
            self.assertTrue(res.success, res.error_message)
            self.assertEqual(asr.call_count, 1)
            self.assertEqual(res.transcript, "Exact cilian speech.")
            self.assertEqual(len(cluster.encoded_calls), 1)
            self.assertEqual(cluster.encoded_calls[0]["transcript"], "Exact cilian speech.")
        finally:
            orch.shutdown()

    def test_cilian_mp3_decode_headroom_and_gentle_mastering(self):
        cilian_path = "/home/sharjeel/Downloads/Ai/History/Wager/characters/Character_voices/cilian_9sec.mp3"
        if not os.path.exists(cilian_path):
            self.skipTest("cilian_9sec.mp3 not present on host")

        pcm, sr = load_any_audio(cilian_path, target_sr=24000, max_duration_s=10.0)
        self.assertEqual(sr, 24000)
        self.assertGreater(len(pcm), 200000)
        # Apply gentle_loudnorm mastering and check headroom is safely < 1.0 without square clipping
        mastered, out_sr = master_audio_array(pcm, sr=sr, preset="gentle_loudnorm")
        self.assertEqual(out_sr, 24000)
        self.assertLessEqual(float(np.max(np.abs(mastered))), 0.95)
        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(mastered ** 2))) + 1e-9)
        self.assertGreater(rms_db, -22.0)
        self.assertLess(rms_db, -12.0)

    def test_enhance_chunking_prevents_tiny_tail_sliver(self):
        # 1) 9.012s audio (e.g., cilian_9sec.mp3) must run single-pass (overlap=0.0) so it doesn't leave a 0.012s sliver
        eff_c, eff_ov, pad_in = _prepare_resemble_chunking(int(9.012 * 24000), 24000, chunk_seconds=10.0, target_rate=44100)
        self.assertEqual(eff_ov, 0.0)
        self.assertEqual(pad_in, 0)
        self.assertGreaterEqual(eff_c, 10.0)

        # 2) Ultra-short 0.25s audio pads cleanly to >= 1.5s
        eff_c_s, eff_ov_s, pad_s = _prepare_resemble_chunking(int(0.25 * 24000), 24000, chunk_seconds=10.0, target_rate=44100)
        self.assertEqual(eff_ov_s, 0.0)
        self.assertGreaterEqual((int(0.25 * 24000) + pad_s) / 24000.0, 1.5)

        # 3) Multi-chunk 18.02s audio (hop=9.0s -> 0.02s sliver) pads tail so final chunk >= 1.5s
        orig_len = int(18.02 * 24000)
        eff_c_m, eff_ov_m, pad_m = _prepare_resemble_chunking(orig_len, 24000, chunk_seconds=10.0, target_rate=44100)
        self.assertEqual(eff_c_m, 10.0)
        self.assertEqual(eff_ov_m, 1.0)
        hop_t = int(44100 * 9.0)
        padded_len_t = int(np.ceil((orig_len + pad_m) * 44100 / 24000.0))
        self.assertGreaterEqual(padded_len_t % hop_t, int(1.5 * 44100))

        # 4) End-to-end check with mocked resemble_enhance (and fake torch if absent locally) preserving exact output length
        import contextlib
        import types
        try:
            import torch  # type: ignore
            need_fake_torch = False
            fake_torch_mods = {}
        except ImportError:
            need_fake_torch = True
            class _FakeTensor:
                def __init__(self, arr):
                    self.arr = np.asarray(arr, dtype=np.float32)
                @property
                def shape(self):
                    return self.arr.shape
                def dim(self):
                    return self.arr.ndim
                def mean(self, dim=-1):
                    return _FakeTensor(self.arr.mean(axis=dim))
                def cpu(self):
                    return self
                def numpy(self):
                    return self.arr
                def __getitem__(self, item):
                    return _FakeTensor(self.arr[item])

            fake_torch = types.ModuleType("torch")
            fake_torch.float32 = np.float32
            fake_torch.from_numpy = lambda a: _FakeTensor(a)
            fake_torch.inference_mode = contextlib.nullcontext
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, device=lambda _i: contextlib.nullcontext(), empty_cache=lambda: None)
            fake_nn = types.ModuleType("torch.nn")
            fake_F = types.ModuleType("torch.nn.functional")
            fake_F.pad = lambda t, pad: _FakeTensor(np.pad(t.arr, pad))
            fake_nn.functional = fake_F
            fake_torch.nn = fake_nn
            fake_torch_mods = {"torch": fake_torch, "torch.nn": fake_nn, "torch.nn.functional": fake_F}
            torch = fake_torch

        fake_pkg = types.ModuleType("resemble_enhance")
        fake_enh_pkg = types.ModuleType("resemble_enhance.enhancer")
        fake_inf_mod = types.ModuleType("resemble_enhance.enhancer.inference")
        observed = {}

        def _fake_inference(model, dwav, sr, device, chunk_seconds=30.0, overlap_seconds=1.0):
            observed["len_in"] = int(dwav.shape[-1])
            observed["chunk_seconds"] = chunk_seconds
            observed["overlap_seconds"] = overlap_seconds
            out_n = int(round(int(dwav.shape[-1]) * 44100.0 / sr))
            out_arr = np.zeros(out_n, dtype=np.float32)
            return (torch.from_numpy(out_arr) if need_fake_torch else torch.zeros(out_n, dtype=torch.float32)), 44100

        fake_inf_mod.load_enhancer = lambda *_a, **_kw: None
        fake_inf_mod.inference = _fake_inference
        fake_enh_pkg.inference = fake_inf_mod
        fake_pkg.enhancer = fake_enh_pkg

        class _DummyModel:
            class _HP:
                wav_rate = 44100
            hp = _HP()
            def to(self, _d):
                return self
            def eval(self):
                return self

        import bz_ggml.audio_pipeline as ap
        prev_model = ap._RESEMBLE_CPU_MODEL
        mod_names = ("resemble_enhance", "resemble_enhance.enhancer", "resemble_enhance.enhancer.inference") + tuple(fake_torch_mods.keys())
        prev_modules = {k: sys.modules.get(k) for k in mod_names}
        try:
            ap._RESEMBLE_CPU_MODEL = _DummyModel()
            for k, v in fake_torch_mods.items():
                sys.modules[k] = v
            sys.modules["resemble_enhance"] = fake_pkg
            sys.modules["resemble_enhance.enhancer"] = fake_enh_pkg
            sys.modules["resemble_enhance.enhancer.inference"] = fake_inf_mod

            pcm_18s = np.zeros(orig_len, dtype=np.float32)
            out_pcm, out_sr = enhance_audio_array(pcm_18s, sr=24000, chunk_seconds=10.0, device="cpu")
            self.assertEqual(out_sr, 44100)
            self.assertEqual(len(out_pcm), int(round(orig_len * 44100.0 / 24000.0)))
            self.assertGreater(observed["len_in"], orig_len)
        finally:
            ap._RESEMBLE_CPU_MODEL = prev_model
            for k, v in prev_modules.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_snr_estimator_and_auto_enhancer_gate(self):
        sr = 24000
        t = np.linspace(0.0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
        clean_speech = 0.25 * np.sin(2.0 * np.pi * 110.0 * t).astype(np.float32)
        # Clean signal with zero pauses must still classify as clean (>= 30 dB) via low highband flatness
        self.assertGreaterEqual(estimate_audio_snr_db(clean_speech, sr), 30.0)

        # Clean signal with quiet pause + high SNR (~46 dB)
        paused = clean_speech.copy()
        paused[: int(0.4 * sr)] *= 0.002
        self.assertGreaterEqual(estimate_audio_snr_db(paused, sr), 30.0)

        # Noisy signal (~10 dB white noise floor) must fall below 30 dB gate
        rng = np.random.default_rng(7)
        noisy = paused + rng.normal(0.0, 0.06, size=len(paused)).astype(np.float32)
        self.assertLess(estimate_audio_snr_db(noisy, sr), 25.0)

        # End-to-end auto_gate test on EngineOrchestrator: clean audio skips Resemble cleanly
        cluster = _DummyVocoderCluster(enable_asr=True, enable_enhancer=False)
        asr = _MockASRService(text="Clean auto gated speech.")
        orch = EngineOrchestrator(
            cluster=cluster,
            asr_service=asr,
            capabilities=EngineCapabilities(enable_enhancer=False, enable_asr=True),
        )
        try:
            p = Parcel.voice_registration(
                voice_name="clean_spk",
                audio_input=paused,
                enhance="auto",
                master=True,
            )
            self.assertEqual([tk.task_type for tk in p.tasks], [TaskType.ENHANCE, TaskType.MASTER, TaskType.ASR, TaskType.VOICE_REGISTER])
            self.assertTrue(p.tasks[0].params.get("auto_gate"))
            res = orch.dispatch(p, timeout=15.0)
            self.assertTrue(res.success, res.error_message)
            self.assertEqual(cluster.encoded_calls[0]["transcript"], "Clean auto gated speech.")
        finally:
            orch.shutdown()

    def test_isolated_chunk_vocoding_and_rms_leveling_stitch(self):
        sr = 24000
        # 3 chunks with large loudness mismatch (-12 dB, -24 dB, -18 dB)
        t = np.linspace(0.0, 0.8, int(0.8 * sr), endpoint=False, dtype=np.float32)
        c0 = (0.35 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
        c1 = (0.09 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
        c2 = (0.18 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)

        stitched = level_and_stitch_chunks([c0, c1, c2], sample_rate=sr, pause_ms=200, max_gain_db=4.0, fade_ms=10.0)
        expected_pause = int(sr * 0.200)
        # Total length equals chunks + two 200ms zero breath gaps
        self.assertEqual(len(stitched), len(c0) + len(c1) + len(c2) + 2 * expected_pause)
        # Verify exact zero pause between chunk 0 and chunk 1
        pause_slice = stitched[len(c0) : len(c0) + expected_pause]
        self.assertTrue(np.all(pause_slice == 0.0))

        # Check isolated RequestTTSAssembly prevents ctx_tokens bleeding across out-of-order chunks
        asm = RequestTTSAssembly(
            request_id="iso_test",
            chunks_text=["c0", "c1", "c2"],
            sample_rate=sr,
            isolate_chunk_vocoding=True,
            pause_ms=200,
            level_chunks=True,
        )
        jobs2 = asm.record_chunk_tokens(2, [2] * 16 * 10)
        self.assertEqual(len(jobs2), 1)
        self.assertEqual(jobs2[0].job_seq, 2)
        self.assertEqual(jobs2[0].ctx_tokens, [])

        jobs0 = asm.record_chunk_tokens(0, [0] * 16 * 10)
        self.assertEqual(jobs0[0].ctx_tokens, [])
        jobs1 = asm.record_chunk_tokens(1, [1] * 16 * 10)
        self.assertTrue(jobs1[0].is_final)

        done, _, _ = asm.record_pcm_part(2, c2)
        self.assertFalse(done)
        done, _, _ = asm.record_pcm_part(0, c0)
        self.assertFalse(done)
        done, full_pcm, merged = asm.record_pcm_part(1, c1)
        self.assertTrue(done)
        self.assertEqual(len(full_pcm), len(stitched))
        self.assertEqual(len(merged), 3 * 16 * 10)

    def test_cpp_sampling_cumulative_repetition_penalty(self):
        import shutil
        import subprocess
        import tempfile

        gxx = shutil.which("g++")
        if not gxx:
            self.skipTest("g++ not available locally")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sampling_cpp = os.path.join(repo_root, "src", "sampling.cpp")
        include_dir = os.path.join(repo_root, "include")

        cpp_driver = r"""
#include "breeze/sampling.h"
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    using namespace breeze;
    SampleParams sp;
    sp.temperature = 0.0f; // deterministic argmax after repetition_penalty
    sp.repetition_penalty = 1.5f;
    std::mt19937 rng(42);

    // 1. Repeated history tokens accumulate penalty (1.5^3 = 3.375x), pushing silence attractors below EOS/speech.
    std::vector<int> hist_rep(3, 5);
    std::vector<float> logits(16, 0.0f);
    logits[5] = 4.0f; // 4.0 / 3.375 = 1.185 < logits[6] = 1.5
    logits[6] = 1.5f;
    int pick1 = sample_token(logits, sp, rng, &hist_rep, nullptr);
    assert(pick1 == 6);

    // 2. Tokens absent from history receive zero penalty.
    std::vector<float> logits2(16, 0.0f);
    logits2[7] = 2.0f;
    logits2[6] = 1.5f;
    int pick2 = sample_token(logits2, sp, rng, &hist_rep, nullptr);
    assert(pick2 == 7);

    std::printf("CPP_SAMPLING_CUMULATIVE_OK\n");
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpd:
            drv = os.path.join(tmpd, "test_samp.cpp")
            exe = os.path.join(tmpd, "test_samp")
            with open(drv, "w") as f:
                f.write(cpp_driver)
            subprocess.run(
                [gxx, "-std=c++17", f"-I{include_dir}", drv, sampling_cpp, "-o", exe],
                check=True,
                capture_output=True,
                text=True,
            )
            res = subprocess.run([exe], check=True, capture_output=True, text=True)
            self.assertIn("CPP_SAMPLING_CUMULATIVE_OK", res.stdout)

    def test_cluster_precision_auto_detection(self):
        from bz_ggml import DualInstanceCluster
        self.assertEqual(DualInstanceCluster.resolve_precision("/models/breeze-tts-2-f16.gguf", None), "fp16")
        self.assertEqual(DualInstanceCluster.resolve_precision("/models/breeze-fp16.gguf", "auto"), "fp16")
        self.assertEqual(DualInstanceCluster.resolve_precision("/models/breeze-tts-2-q8_0.gguf", "auto"), "int8")
        self.assertEqual(DualInstanceCluster.resolve_precision("/models/breeze-tts-2-f16.gguf", "int8"), "int8")

    def test_gateway_parse_dispatch_job_auto_gate_and_explicit_enhance(self):
        import json
        from types import SimpleNamespace
        from bz_ggml.gateway import MTLSGatewayClient

        gw = MTLSGatewayClient.__new__(MTLSGatewayClient)

        d_vreg = SimpleNamespace(
            job_type_name="voice_registration",
            quality="standard",
            params=json.dumps({"voice_name": "cilian", "audio_input": "/tmp/sample.wav"}).encode("utf-8"),
        )
        p_vreg = gw._parse_dispatch_job("job_vreg", d_vreg)
        types_vreg = [t.task_type for t in p_vreg.tasks]
        self.assertEqual(types_vreg, [TaskType.ENHANCE, TaskType.MASTER, TaskType.ASR, TaskType.VOICE_REGISTER])
        self.assertEqual(p_vreg.tasks[0].params.get("auto_gate"), True)
        self.assertEqual(p_vreg.tasks[0].params.get("min_snr_db"), 30.0)

        d_studio_raw = SimpleNamespace(
            job_type_name="tts_clone",
            quality="studio",
            params=json.dumps({"script_text": "hello", "enhance": False, "master": True}).encode("utf-8"),
        )
        p_studio_raw = gw._parse_dispatch_job("job_studio_raw", d_studio_raw)
        types_raw = [t.task_type for t in p_studio_raw.tasks]
        self.assertEqual(types_raw, [TaskType.TTS, TaskType.MASTER])

        d_studio_def = SimpleNamespace(
            job_type_name="tts_clone",
            quality="studio",
            params=json.dumps({"script_text": "hello"}).encode("utf-8"),
        )
        p_studio_def = gw._parse_dispatch_job("job_studio_def", d_studio_def)
        self.assertEqual([t.task_type for t in p_studio_def.tasks], [TaskType.TTS, TaskType.ENHANCE, TaskType.MASTER])

    def test_gradio_app_option_a_defaults_and_helpers(self):
        repo_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
        if repo_python not in sys.path:
            sys.path.insert(0, repo_python)
        try:
            import gradio_app
        except ImportError:
            self.skipTest("gradio not installed locally; tested remotely on Kaggle")

        self.assertEqual(gradio_app.DEFAULT_OPTION_A_INSTRUCTION, "Speak clearly and naturally.")
        self.assertEqual(gradio_app._resolve_reg_enhance_mode(gradio_app.REG_ENHANCE_CHOICES[0])[0], "auto")
        self.assertIs(gradio_app._resolve_reg_enhance_mode(gradio_app.REG_ENHANCE_CHOICES[1])[0], True)
        self.assertIs(gradio_app._resolve_reg_enhance_mode(gradio_app.REG_ENHANCE_CHOICES[2])[0], False)
        demo = gradio_app.build_app()
        self.assertIsNotNone(demo)

        # Verify UI handler signatures & output tuple arities with mock orchestrator
        import tempfile
        from bz_ggml import ParcelResult, serialize_breeze_voice
        orig_get_orch = gradio_app.get_orchestrator
        orig_cfg = dict(gradio_app.APP_CONFIG)
        with tempfile.TemporaryDirectory() as tmpd:
            try:
                gradio_app.APP_CONFIG["voices_dir"] = tmpd
                gradio_app.APP_CONFIG["outputs_dir"] = tmpd
                fake_voice_bytes = serialize_breeze_voice(
                    codes=[10] * (16 * 50), frames=50, text="mock sample", sample_rate=24000
                )

                class MockOrch:
                    def __init__(self):
                        self.parcels = []

                    def dispatch(self, parcel, progress_callback=None):
                        self.parcels.append(parcel)
                        if any(t.task_type == TaskType.VOICE_REGISTER for t in parcel.tasks):
                            return ParcelResult(
                                parcel_id=parcel.parcel_id,
                                success=True,
                                voice_bytes=fake_voice_bytes,
                                transcript="mock sample",
                                stage_timings={"enhance": 0.0, "asr": 0.1},
                            )
                        pcm = np.zeros(24000, dtype=np.float32)
                        return ParcelResult(
                            parcel_id=parcel.parcel_id,
                            success=True,
                            duration_s=1.0,
                            chunks_count=1,
                            sample_rate=24000,
                            audio_pcm=pcm,
                            raw_audio_pcm=pcm,
                            subtitles_srt="1\n00:00:00,000 --> 00:00:01,000\nmock\n",
                            stage_timings={"tts": 0.1},
                        )

                mock_orch = MockOrch()
                gradio_app.get_orchestrator = lambda: mock_orch
                fake_wav = os.path.join(tmpd, "ref.wav")
                with open(fake_wav, "wb") as wf:
                    wf.write(b"RIFF")

                reg_res = gradio_app.register_voice_file(
                    "mock_spk", fake_wav, gradio_app.REG_ENHANCE_CHOICES[0], ""
                )
                self.assertEqual(len(reg_res), 6)
                self.assertTrue(os.path.exists(os.path.join(tmpd, "mock_spk.breeze")))
                self.assertEqual(mock_orch.parcels[-1].tasks[0].params.get("auto_gate"), True)

                tts_res = list(gradio_app.tts_generate(
                    "Hello world.", gradio_app.DEFAULT_OPTION_A_INSTRUCTION, "(None / Default Voice)",
                    1.6, 0.55, 0.85, 30, 1.08, 0.0, 3, 42
                ))[-1]
                self.assertEqual(len(tts_res), 3)
                self.assertTrue(os.path.exists(tts_res[0]))

                clone_res = list(gradio_app.voice_clone_generate(
                    "Hello cloned world.", "Mock Spk", gradio_app.DEFAULT_OPTION_A_INSTRUCTION,
                    1.6, 0.55, 0.85, 30, 1.08, 0.0, 3, 42
                ))[-1]
                self.assertEqual(len(clone_res), 3)
                self.assertEqual(mock_orch.parcels[-1].payload["depth_cfg_codebooks"], 3)
                self.assertEqual(mock_orch.parcels[-1].payload["depth_temperature"], 0.0)
                self.assertEqual(mock_orch.parcels[-1].payload["cfg_scale"], 1.6)

                lf_res = list(gradio_app.run_longform_pipeline_ui(
                    "Longform sentence one. Longform sentence two.",
                    "Mock Spk",
                    gradio_app.DEFAULT_OPTION_A_INSTRUCTION,
                    1.6, 0.55, 0.85, 30, 1.08, 0.0, 3, 42, 200,
                    True, 45, 60, False, True, True
                ))[-1]
                self.assertEqual(len(lf_res), 8)
                self.assertIsNone(mock_orch.parcels[-1].payload["target_words"])
                self.assertIsNone(mock_orch.parcels[-1].payload["max_words"])
                self.assertEqual(
                    [t.task_type for t in mock_orch.parcels[-1].tasks],
                    [TaskType.TTS, TaskType.MASTER, TaskType.ASR],
                )
            finally:
                gradio_app.get_orchestrator = orig_get_orch
                gradio_app.APP_CONFIG.update(orig_cfg)


if __name__ == "__main__":
    unittest.main()
