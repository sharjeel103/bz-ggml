import asyncio
import io
import json
import logging
import os
import sys
import time
import unittest
import numpy as np
import scipy.io.wavfile

# Ensure python/ is on sys.path
for candidate in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Downloads", "coding", "bz-ggml", "python")),
    "/kaggle/working/bz-ggml/python"
]:
    if os.path.exists(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from bz_ggml.audio_pipeline import (
    load_any_audio,
    encode_pcm_to_wav_bytes,
    encode_pcm_to_mp3_bytes,
    apply_studio_normalization,
    stitch_audio_chunks
)
from bz_ggml.orchestrator import (
    EngineOrchestrator,
    Parcel,
    TaskType,
    TaskSpec,
    ParcelResult,
    StageJob,
    RequestTTSAssembly
)
from bz_ggml.gateway import (
    AdmissionController,
    TelemetrySampler,
    MTLSGatewayClient
)
from bz_ggml.proto import worker_pb2, engine_pb2

logging.basicConfig(level=logging.INFO)

class MockCluster:
    sample_rate = 24000
    dynamic_coord = type("MockCoord", (), {
        "active_tokens": {0: 1024, 1: 512},
        "get_admission_ceiling": lambda self, dev: 8192
    })()


class TestSymmetricArchitecture(unittest.TestCase):

    def test_01_audio_trimming_and_loading(self):
        print("\n--- TEST 1: Universal Audio Ingestion & Hard 10s Clipping ---")
        # Create synthetic 30-second stereo audio at 48000 Hz
        sr_orig = 48000
        duration_orig = 30.0
        t = np.linspace(0, duration_orig, int(sr_orig * duration_orig), endpoint=False)
        tone_left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        tone_right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo_pcm = np.stack([tone_left, tone_right], axis=1)

        # Write to in-memory WAV
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sr_orig, (stereo_pcm * 32767).astype(np.int16))
        wav_bytes = buf.getvalue()

        # Ingest with load_any_audio: target_sr=24000, max_duration_s=10.0
        loaded_pcm, sr_out = load_any_audio(wav_bytes, target_sr=24000, max_duration_s=10.0)

        self.assertEqual(sr_out, 24000)
        self.assertEqual(loaded_pcm.ndim, 1)  # Converted to mono
        max_allowed_samples = int(10.0 * 24000)
        self.assertLessEqual(len(loaded_pcm), max_allowed_samples)
        self.assertEqual(len(loaded_pcm), max_allowed_samples)
        print(f"✓ Ingested 30s stereo audio, resampled to {sr_out}Hz mono, hard-clipped to exactly {len(loaded_pcm)} samples ({len(loaded_pcm)/sr_out:.2f}s).")

    def test_02_pcm_encoding_formats(self):
        print("\n--- TEST 2: In-Memory Audio Byte Encoders (WAV & MP3) ---")
        sr = 24000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        wav_bytes = encode_pcm_to_wav_bytes(pcm, sr)
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertGreater(len(wav_bytes), 44)
        print(f"✓ WAV bytes encoded cleanly: {len(wav_bytes)} bytes with valid RIFF header.")

        try:
            mp3_bytes = encode_pcm_to_mp3_bytes(pcm, sr, bitrate="192k")
            self.assertGreater(len(mp3_bytes), 1000)
            print(f"✓ MP3 bytes encoded cleanly via FFmpeg: {len(mp3_bytes)} bytes.")
        except RuntimeError as e:
            print(f"Note: FFmpeg not available or skipped for MP3 encoding: {e}")

    def test_03_admission_controller_heuristics_and_bounds(self):
        print("\n--- TEST 3: Deterministic Admission Controller & Dedicated Queue Ceilings ---")
        orch = EngineOrchestrator(cluster=MockCluster())
        admission = AdmissionController(orch, max_queue_wait_s=60.0)

        # 1. Estimate durations
        parcel_tts = Parcel(
            parcel_id="p1",
            tasks=[TaskSpec(TaskType.TTS)],
            payload={"script_text": "This is a test script with eight words in total."}
        )
        dur_tts = admission.estimate_parcel_duration(parcel_tts)
        self.assertGreater(dur_tts, 0.4)

        parcel_voice_reg = Parcel.voice_registration("v_test", b"dummy", enhance=True)
        dur_voice_reg = admission.estimate_parcel_duration(parcel_voice_reg)
        # Should include ENHANCE (6.5s) + ASR (0.20s) + VOICE_REGISTER (1.2s) = ~7.9s
        self.assertGreaterEqual(dur_voice_reg, 7.0)
        print(f"✓ Duration heuristics: TTS (8 words) = {dur_tts}s | Full Voice Reg Pipeline = {dur_voice_reg}s")

        # 2. Test admission evaluation when queues are empty
        accepted, reason, est_dur, eta, pos = admission.evaluate_admission(parcel_voice_reg)
        self.assertTrue(accepted)
        self.assertEqual(pos, 1)
        self.assertAlmostEqual(eta, dur_voice_reg, places=1)

        # 3. Fill enhance queue to maximum ceiling (8 items) and test stage-specific overflow rejection
        for i in range(EngineOrchestrator.MAX_QUEUE_ENHANCE):
            orch.enhance_queue.put(StageJob(job_id=f"mock_enh_{i}", request_id=f"r_{i}", task_type=TaskType.ENHANCE))

        accepted_overflow, reason_overflow, _, _, _ = admission.evaluate_admission(parcel_voice_reg)
        self.assertFalse(accepted_overflow)
        self.assertIn("REJECTED_STAGE_OVERFLOW", reason_overflow)
        self.assertIn("enhance", reason_overflow)
        print(f"✓ Stage-specific backpressure overflow rejection validated: '{reason_overflow}'")

        # 4. Clean up queue and test pre-cancellation
        while not orch.enhance_queue.empty():
            orch.enhance_queue.get_nowait()

        admission.cancelled_job_ids.clear()
        admission.cancelled_job_ids.add("p_cancelled")
        parcel_canc = Parcel(parcel_id="p_cancelled", tasks=[TaskSpec(TaskType.TTS)], payload={"script_text": "hello"})
        acc_canc, reason_canc, _, _, _ = admission.evaluate_admission(parcel_canc)
        self.assertFalse(acc_canc)
        self.assertIn("REJECTED_CANCELLED", reason_canc)
        print(f"✓ Pre-cancellation validated: '{reason_canc}'")

    def test_04_telemetry_sampler_low_overhead(self):
        print("\n--- TEST 4: Microsecond Telemetry Sampler Benchmark ---")
        orch = EngineOrchestrator(cluster=MockCluster())
        sampler = TelemetrySampler("test-node-01", orch)

        # Sample 100 times and measure execution latency
        t0 = time.perf_counter()
        N = 100
        for _ in range(N):
            metrics = sampler.sample_metrics()
        elapsed = time.perf_counter() - t0
        avg_latency_us = (elapsed / N) * 1_000_000

        self.assertIn("cpu_percent", metrics)
        self.assertIn("ram_used_gb", metrics)
        self.assertIn("gpu0_token_ceiling", metrics)
        self.assertIn("queue_depths", metrics)
        self.assertIn("tts", metrics["queue_depths"])
        self.assertLess(avg_latency_us, 500.0)  # Must be under 500 microseconds
        print(f"✓ Telemetry sampling average latency: {avg_latency_us:.2f} µs per sample (< 0.1% CPU core overhead).")

    def test_05_workerpb_wire_contract_serialization(self):
        print("\n--- TEST 5: Official workerpb Protobuf Envelope Wire Contract ---")
        # 1. RegisterRequest
        reg_req = worker_pb2.RegisterRequest(
            gpu_count=2,
            vram_gb=32,
            job_types=[
                worker_pb2.JobTypeCapability(
                    job_type_name="tts_clone",
                    quality_tiers=["standard", "studio"],
                    max_concurrent=20,
                    throughput_unit_per_sec=18.0
                )
            ],
            global_max_concurrent=20,
            declared_lease_seconds=604800
        )
        env = worker_pb2.Envelope(
            job_id="",
            attempt=0,
            type=worker_pb2.JobType.REGISTER,
            payload=reg_req.SerializeToString(),
            ts_unix_ms=int(time.time() * 1000)
        )
        wire_bytes = env.SerializeToString()
        self.assertGreater(len(wire_bytes), 20)

        # Unmarshal
        env_recv = worker_pb2.Envelope()
        env_recv.ParseFromString(wire_bytes)
        self.assertEqual(env_recv.type, worker_pb2.JobType.REGISTER)
        reg_unpacked = worker_pb2.RegisterRequest()
        reg_unpacked.ParseFromString(env_recv.payload)
        self.assertEqual(reg_unpacked.gpu_count, 2)
        self.assertEqual(reg_unpacked.vram_gb, 32)
        self.assertEqual(reg_unpacked.job_types[0].job_type_name, "tts_clone")
        print(f"✓ workerpb.REGISTER envelope serialized & deserialized ({len(wire_bytes)} bytes).")

        # 2. DispatchJob & AckJob / DenyJob
        dispatch = worker_pb2.DispatchJob(
            job_type_name="tts_clone",
            quality="studio",
            params=json.dumps({"script_text": "Hello world"}).encode("utf-8"),
            priority=1
        )
        disp_env = worker_pb2.Envelope(
            job_id="job-abc-123",
            attempt=2,  # Must echo attempt back
            type=worker_pb2.JobType.DISPATCH,
            payload=dispatch.SerializeToString(),
            ts_unix_ms=int(time.time() * 1000)
        )

        ack = worker_pb2.AckJob(eta_seconds=4.5)
        ack_env = worker_pb2.Envelope(
            job_id=disp_env.job_id,
            attempt=disp_env.attempt,
            type=worker_pb2.JobType.ACK,
            payload=ack.SerializeToString(),
            ts_unix_ms=int(time.time() * 1000)
        )
        self.assertEqual(ack_env.job_id, "job-abc-123")
        self.assertEqual(ack_env.attempt, 2)

        # 3. Status
        status = worker_pb2.JobStatus(state=worker_pb2.JobStatus.COMPLETED, error="")
        status_env = worker_pb2.Envelope(
            job_id=disp_env.job_id,
            attempt=disp_env.attempt,
            type=worker_pb2.JobType.STATUS,
            payload=status.SerializeToString(),
            ts_unix_ms=int(time.time() * 1000)
        )
        self.assertEqual(status_env.attempt, 2)
        print("✓ Fencing attempt echoed faithfully across DISPATCH -> ACK -> STATUS.")

    def test_06_direct_cloud_upload_validation_and_fallback(self):
        print("\n--- TEST 6: Atomic Cloud Upload & Transactional Safety Fallback ---")
        orch = EngineOrchestrator(cluster=MockCluster())

        # Test pre-flight URL validation: invalid URL must fail validation
        parcel_invalid_url = Parcel(
            parcel_id="p_bad_url",
            tasks=[TaskSpec(TaskType.UPLOAD, {"targets": {"audio_mp3": "ftp://not-allowed.com/file"}})],
            upload_targets={"audio_mp3": "ftp://not-allowed.com/file"}
        )
        with self.assertRaises(ValueError) as ctx:
            orch.validate_parcel(parcel_invalid_url)
        print(f"✓ Pre-flight URL validator blocked non-HTTP scheme: '{ctx.exception}'")

        # Test fallback on failed upload target
        import http.server
        import socketserver
        import threading

        class ForbiddenHandler(http.server.BaseHTTPRequestHandler):
            def do_PUT(self):
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden")
            def log_message(self, format, *args):
                pass

        server = socketserver.TCPServer(("127.0.0.1", 0), ForbiddenHandler)
        port = server.server_address[1]
        t_server = threading.Thread(target=server.serve_forever, daemon=True)
        t_server.start()

        put_url = f"http://127.0.0.1:{port}/audio.mp3"
        fake_pcm = np.zeros(24000, dtype=np.float32)

        parcel = Parcel(
            parcel_id="p_upload_test",
            tasks=[TaskSpec(TaskType.UPLOAD)],
            payload={
                "audio_input": fake_pcm,
                "upload_targets": {"audio_mp3": put_url}
            },
            upload_targets={"audio_mp3": put_url}
        )

        # Dispatch via orchestrator
        res = orch.dispatch(parcel)
        server.shutdown()

        self.assertIn("audio_mp3", res.upload_results)
        self.assertFalse(res.upload_results["audio_mp3"]["success"])
        # Transactional fallback: artifact_bytes MUST retain raw audio buffer so zero compute is lost
        self.assertIn("audio_mp3", res.artifact_bytes)
        self.assertGreater(len(res.artifact_bytes["audio_mp3"]), 0)
        print(f"✓ Zero-lost-compute transactional fallback verified: HTTP 403 captured, raw buffer ({len(res.artifact_bytes['audio_mp3'])} bytes) safely preserved.")

    def test_07_autonomous_tts_chunking_and_out_of_order_assembly(self):
        print("\n--- TEST 7: Autonomous TTS Subsystem: Chunking Rule & Asynchronous Out-of-Order Assembly ---")
        orch = EngineOrchestrator(cluster=MockCluster())

        # 1. 2-Minute / 300-word rule test
        short_script = "This is a short script with only ten simple words total."
        long_script = " ".join(["The quick brown fox jumps over the lazy dog repeatedly."] * 35) # ~315 words

        short_chunks = orch._prepare_tts_chunks(short_script)
        long_chunks = orch._prepare_tts_chunks(long_script)

        self.assertEqual(len(short_chunks), 1, "Scripts <= 280 words must run as 1 chunk")
        self.assertGreater(len(long_chunks), 1, "Scripts > 280 words must be split into semantic chunks")
        print(f"✓ Elastic Chunking: Short script ({len(short_script.split())} words) -> {len(short_chunks)} chunk | Long script ({len(long_script.split())} words) -> {len(long_chunks)} chunks.")

        # 2. Elastic paragraph lookahead partitioning (Two ~170-word paragraphs)
        p1 = " ".join(["The voice of an author carries distinct rhythm, timbre, and intention through sentences."] * 14)
        p2 = " ".join(["When speech synthesis models respect semantic boundaries, paragraphs flow naturally without clipping."] * 13)
        two_para_script = f"{p1}\n\n{p2}"
        para_chunks = orch._prepare_tts_chunks(two_para_script)
        self.assertEqual(len(para_chunks), 2, "Two paragraphs should cleanly partition into 2 chunks at the paragraph boundary")
        print(f"✓ Elastic Paragraph Lookahead: 337-word 2-paragraph script cleanly split into exactly {len(para_chunks)} chunks: {[len(c.split()) for c in para_chunks]} words.")

        # 3. Out-of-order chunk assembly simulation
        assembly = RequestTTSAssembly(request_id="req_multi_chunk", total_chunks=3, sample_rate=24000, pause_ms=200)

        # Chunk 0: 1 second 440 Hz
        # Chunk 1: 1 second 880 Hz
        # Chunk 2: 1 second 1320 Hz
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        pcm_0 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        pcm_1 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        pcm_2 = np.sin(2 * np.pi * 1320 * t).astype(np.float32)

        # Chunks arrive out of order: Chunk 1 arrives first!
        res_after_1 = assembly.add_chunk(1, pcm_1)
        self.assertIsNone(res_after_1, "Assembly must return None until ALL chunks arrive")

        # Chunk 2 arrives second
        res_after_2 = assembly.add_chunk(2, pcm_2)
        self.assertIsNone(res_after_2, "Assembly must return None until ALL chunks arrive")

        # Chunk 0 arrives last
        stitched_pcm = assembly.add_chunk(0, pcm_0)
        self.assertIsNotNone(stitched_pcm, "Assembly must stitch and return final PCM once all chunks arrive")

        # Total expected samples: 3 * 24000 + 2 * (0.2s * 24000) = 72000 + 9600 = 81600 samples
        expected_samples = 3 * sr + 2 * int(0.2 * sr)
        self.assertEqual(len(stitched_pcm), expected_samples)
        print(f"✓ Out-of-order assembly succeeded: [Chunk 1, Chunk 2, Chunk 0] assembled chronologically to {len(stitched_pcm)} samples with 200ms breath pauses.")

    def test_08_decoupled_queues_and_pure_routing(self):
        print("\n--- TEST 8: Decoupled Stage Queues & Permutation Direct Routing ---")
        orch = EngineOrchestrator(cluster=MockCluster())

        # Test queue ceilings
        self.assertEqual(orch.tts_queue.maxsize, 150)
        self.assertEqual(orch.enhance_queue.maxsize, 8)
        self.assertEqual(orch.asr_queue.maxsize, 20)
        self.assertEqual(orch.voice_reg_queue.maxsize, 15)
        self.assertEqual(orch.master_queue.maxsize, 20)
        self.assertEqual(orch.upload_queue.maxsize, 100)
        print("✓ All 6 stage queues verified with exact hardware-calibrated ceilings.")

        # Test Direct Routing of permutations:
        # Enhance-only parcel
        fake_pcm = np.zeros(24000, dtype=np.float32)
        parcel_enh = Parcel(
            parcel_id="p_enh_only",
            tasks=[TaskSpec(TaskType.ENHANCE)],
            payload={"audio_input": fake_pcm}
        )
        req_enh = orch._create_request_context(parcel_enh)
        self.assertEqual(req_enh.remaining_tasks[0].task_type, TaskType.ENHANCE)

        # Enqueue first task
        orch._enqueue_next_task(req_enh)
        self.assertEqual(orch.enhance_queue.qsize(), 1, "Enhance-only parcel must route directly to enhance_queue")
        job = orch.enhance_queue.get_nowait()
        self.assertEqual(job.task_type, TaskType.ENHANCE)
        self.assertEqual(job.request_id, "p_enh_only")
        print("✓ Enhance-only parcel routed directly into enhance_queue without TTS or monolithic loop.")

        # ASR-only parcel
        parcel_asr = Parcel(
            parcel_id="p_asr_only",
            tasks=[TaskSpec(TaskType.ASR)],
            payload={"audio_input": fake_pcm}
        )
        req_asr = orch._create_request_context(parcel_asr)
        orch._enqueue_next_task(req_asr)
        self.assertEqual(orch.asr_queue.qsize(), 1, "ASR-only parcel must route directly to asr_queue")
        job_asr = orch.asr_queue.get_nowait()
        self.assertEqual(job_asr.task_type, TaskType.ASR)
        self.assertEqual(job_asr.request_id, "p_asr_only")
        print("✓ ASR-only parcel routed directly into asr_queue without other stages.")

if __name__ == "__main__":
    unittest.main()
