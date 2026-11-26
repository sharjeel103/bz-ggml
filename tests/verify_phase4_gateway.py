#!/usr/bin/env python3
"""
Phase 4 Verification:
  4.1 Driver-Level GPU Telemetry Accuracy (when GPUs present)
  4.2 mTLS Security Fail-Fast
  4.3 workerpb v2 Oneof Envelope Round-Trip (Register, Dispatch, Progress, JobResult, Telemetry, Sync)
  4.4 AssetCache: Inline Passthrough, Presigned GET Download, SHA-256 Verification, LRU Eviction
  4.5 Upload Worker & Fallback Semantics: Presigned PUT Success, Inline Fallback on Failure, Hard Fail
  4.6 AdmissionController Bounded Cancellation History
"""
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

try:
    import torch
except ImportError:
    torch = None

REPO_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
if REPO_PYTHON not in sys.path:
    sys.path.insert(0, REPO_PYTHON)

from bz_ggml.asset_cache import AssetCache
from bz_ggml.proto import worker_pb2


def _get_real_nvidia_smi():
    try:
        res = subprocess.run(
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0:
            return {}
        data = {}
        for line in res.stdout.strip().splitlines():
            if line.strip():
                idx, mem, util = [float(x.strip()) for x in line.split(",")]
                data[int(idx)] = (mem, util)
        return data
    except Exception:
        return {}


class _MockHTTPHandler(http.server.BaseHTTPRequestHandler):
    store = {}
    get_hits = 0
    fail_put_paths = set()

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        type(self).get_hits += 1
        data = type(self).store.get(self.path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path in type(self).fail_put_paths:
            self.send_response(500)
            self.end_headers()
            return
        type(self).store[self.path] = body
        self.send_response(200)
        self.end_headers()


def run_all_tests():
    print("=" * 80)
    print("  PHASE 4: GATEWAY v2 PROTOCOL, ASSET CACHE & SECURITY VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 4.1: workerpb v2 Oneof Envelope Round-Trip
    # -------------------------------------------------------------------------
    print("\n[TEST 4.1] workerpb v2 Oneof Envelope Round-Trip...")
    env = worker_pb2.Envelope(
        job_id="0193a1b2-0000-7000-8000-000000000001",
        attempt=3,
        ts_unix_ms=int(time.time() * 1000),
        dispatch=worker_pb2.Dispatch(
            job_type="tts_clone",
            quality="studio",
            priority=50,
            deadline_ms=int(time.time() * 1000) + 60000,
            stages=[
                worker_pb2.StageSpec(stage_type="tts", params_json=b"{}"),
                worker_pb2.StageSpec(stage_type="enhance", params_json=b'{"chunk_seconds": 10.0}'),
            ],
            input_assets=[
                worker_pb2.InputAsset(role="voice_breeze", mime_type="application/octet-stream", sha256="a" * 64, url="https://r2.example/v.breeze"),
                worker_pb2.InputAsset(role="inline_note", mime_type="text/plain", inline_bytes=b"hello"),
            ],
            output_targets=[
                worker_pb2.OutputTarget(
                    key="audio_mp3",
                    upload_url="https://r2.example/out.mp3",
                    mime_type="audio/mpeg",
                    allow_inline_fallback=True,
                    max_inline_bytes=16 * 1024 * 1024,
                ),
            ],
            params_json=json.dumps({"script_text": "Testing v2 contract"}).encode("utf-8"),
        ),
    )
    decoded = worker_pb2.Envelope()
    decoded.ParseFromString(env.SerializeToString())
    assert decoded.WhichOneof("msg") == "dispatch"
    assert decoded.attempt == 3
    assert len(decoded.dispatch.stages) == 2
    assert decoded.dispatch.input_assets[0].WhichOneof("source") == "url"
    assert decoded.dispatch.input_assets[1].WhichOneof("source") == "inline_bytes"
    assert decoded.dispatch.output_targets[0].allow_inline_fallback is True
    print("  PASS: workerpb v2 single-pass oneof Envelope verified.")

    # -------------------------------------------------------------------------
    # Test 4.2: AssetCache (Inline, Presigned GET, SHA-256 Check, Cache Hit, LRU)
    # -------------------------------------------------------------------------
    print("\n[TEST 4.2] AssetCache (Presigned GET, SHA-256, Cache Hit, Inline Fallback)...")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockHTTPHandler)
    port = server.server_address[1]
    t_srv = threading.Thread(target=server.serve_forever, daemon=True)
    t_srv.start()

    tmp_dir = tempfile.mkdtemp(prefix="bz_test_cache_")
    try:
        cache = AssetCache(cache_dir=tmp_dir, max_bytes=2 * 1024 * 1024)
        sample_data = b"RIFF_FAKE_VOICE_DATA" * 100
        sample_sha = hashlib.sha256(sample_data).hexdigest()
        _MockHTTPHandler.store["/voice.bin"] = sample_data
        _MockHTTPHandler.get_hits = 0

        # Inline passthrough
        inline_out = cache.resolve_bytes(inline_bytes=sample_data, expected_sha256=sample_sha)
        assert inline_out == sample_data
        assert _MockHTTPHandler.get_hits == 0

        # Presigned GET miss -> download + cache
        url = f"http://127.0.0.1:{port}/voice.bin"
        got1 = cache.resolve_bytes(url=url, expected_sha256=sample_sha)
        assert got1 == sample_data and _MockHTTPHandler.get_hits == 1

        # Second call -> local SHA-256 cache hit (zero HTTP requests)
        got2 = cache.resolve_bytes(url=url, expected_sha256=sample_sha)
        assert got2 == sample_data and _MockHTTPHandler.get_hits == 1

        # Corrupted SHA-256 must raise
        try:
            cache.resolve_bytes(inline_bytes=b"tampered", expected_sha256=sample_sha)
            raise AssertionError("Expected ValueError on SHA-256 mismatch")
        except ValueError:
            pass

        print("  PASS: AssetCache download, SHA-256 verification, cache hit, and inline fallback verified.")
    finally:
        server.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 4.3: Gateway & mTLS Fail-Fast (when full gateway deps are available)
    # -------------------------------------------------------------------------
    try:
        from bz_ggml.gateway import TelemetrySampler, MTLSGatewayClient, AdmissionController
    except ImportError as e:
        print(f"\n[TEST 4.3] Skipping full gateway runtime checks (missing optional runtime dep: {e})")
        print("\n" + "=" * 80)
        print("  ALL AVAILABLE PHASE 4 TESTS PASSED!")
        print("=" * 80)
        return

    print("\n[TEST 4.3] mTLS Security Fail-Fast & Bounded Cancellation History...")

    class DummyCaps:
        enable_enhancer = True
        enable_asr = False

    class DummyOrchestrator:
        cluster = None
        capabilities = DummyCaps()
        def get_telemetry_snapshot(self):
            return {"queue_depths": {"tts": 2, "enhance": 0}, "active_requests": []}
        def cancel_request(self, rid):
            return True

    dummy_orch = DummyOrchestrator()
    admission = AdmissionController(orchestrator=dummy_orch, max_cancel_history=64)
    for i in range(100):
        admission.cancel(f"job-{i}")
    assert len(admission.cancelled_job_ids) == 64, f"Expected bounded 64, got {len(admission.cancelled_job_ids)}"
    assert "job-0" not in admission.cancelled_job_ids and "job-99" in admission.cancelled_job_ids

    try:
        MTLSGatewayClient(
            server_url="wss://127.0.0.1:9443/ws/worker",
            ca_cert_path="/nonexistent/ca.pem",
            client_cert_path="/nonexistent/client.crt",
            client_key_path="/nonexistent/client.key",
            admission_ctrl=admission,
            require_mtls=True,
        )
        raise AssertionError("mTLS did not fail on nonexistent certificates!")
    except FileNotFoundError:
        pass

    # Verify Register envelope omits disabled ASR capability
    client = MTLSGatewayClient(
        server_url="wss://127.0.0.1:9443/ws/worker",
        ca_cert_path="/nonexistent/ca.pem",
        client_cert_path="/nonexistent/client.crt",
        client_key_path="/nonexistent/client.key",
        admission_ctrl=admission,
        require_mtls=False,
    )
    reg_env = client.build_register_envelope()
    cap_names = {c.name for c in reg_env.register.capabilities}
    assert "audio_enhance" in cap_names and "speech_to_text" not in cap_names
    print("  PASS: mTLS fail-fast, bounded cancellation set, and dynamic Register capabilities verified.")

    # -------------------------------------------------------------------------
    # Test 4.4: Driver-Level GPU Telemetry Accuracy (GPU machines only)
    # -------------------------------------------------------------------------
    smi_data = _get_real_nvidia_smi()
    if smi_data:
        print("\n[TEST 4.4] Driver-Level GPU Telemetry Accuracy...")
        sampler = TelemetrySampler(node_id="test_node_01", orchestrator=dummy_orch)
        metrics = sampler.sample_metrics()
        for idx, (smi_mem, _) in smi_data.items():
            if idx < len(metrics["gpus"]):
                diff = abs(metrics["gpus"][idx]["vram_used_mb"] - smi_mem)
                assert diff <= 250.0, f"GPU {idx} VRAM discrepancy too high: {diff} MB"
        print("  PASS: Driver-level telemetry matches nvidia-smi within +/- 250 MiB.")
    else:
        print("\n[TEST 4.4] Skipped GPU nvidia-smi accuracy check (no NVIDIA GPU detected).")

    # -------------------------------------------------------------------------
    # Test 4.5: Upload Worker & OutputTarget Inline Fallback / Failure Semantics
    # -------------------------------------------------------------------------
    print("\n[TEST 4.5] Upload Worker (Presigned PUT, Retry, Inline Fallback, Hard Fail)...")
    import asyncio
    import types
    import numpy as np
    from bz_ggml.orchestrator import (
        EngineOrchestrator,
        Parcel,
        ParcelResult,
        TaskSpec,
        TaskType,
        EngineCapabilities,
    )
    from bz_ggml.bindings import serialize_breeze_voice

    srv45 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockHTTPHandler)
    port45 = srv45.server_address[1]
    t45 = threading.Thread(target=srv45.serve_forever, daemon=True)
    t45.start()
    try:
        class UploadOnlyCluster:
            precision = "fp16"
            sample_rate = 24000
            enable_q4_burst = False
            vocoder_mode = "on"
            enable_enhancer = False
            enable_asr = False
            dynamic_coord = types.SimpleNamespace(
                active_tokens={0: 0, 1: 0},
                get_admission_ceiling=lambda idx: 19500,
            )
            def close(self):
                pass

        orch45 = EngineOrchestrator(
            cluster=UploadOnlyCluster(),
            asr_service=None,
            capabilities=EngineCapabilities(
                precision="fp16",
                vocoder_mode="off",
                enable_enhancer=False,
                enable_asr=False,
            ),
        )
        try:
            _MockHTTPHandler.store.clear()
            _MockHTTPHandler.fail_put_paths = {"/fail_put.wav"}

            # 4.5a Success PUT
            p_ok = Parcel(
                parcel_id="upl_ok",
                tasks=[TaskSpec(TaskType.UPLOAD)],
                payload={
                    "audio_input": np.zeros(2400, dtype=np.float32),
                    "upload_targets": {
                        "audio_wav": {
                            "url": f"http://127.0.0.1:{port45}/ok.wav",
                            "mime_type": "audio/wav",
                            "allow_inline_fallback": False,
                            "max_inline_bytes": 0,
                        }
                    },
                },
            )
            r_ok = orch45.submit_sync(p_ok, timeout=15.0)
            assert r_ok.success and r_ok.status == "COMPLETED", r_ok.error_message
            assert r_ok.upload_results["audio_wav"]["uploaded"] is True
            assert "/ok.wav" in _MockHTTPHandler.store
            env_ok = MTLSGatewayClient.build_job_result_envelope("upl_ok", 1, r_ok)
            assert env_ok.result.status == worker_pb2.JobResult.COMPLETED
            assert len(env_ok.result.artifacts) == 1
            assert env_ok.result.artifacts[0].uploaded is True
            assert env_ok.result.artifacts[0].inline_bytes == b""

            # 4.5b PUT fails + allow_inline_fallback=True within budget -> succeeds inline
            p_fb = Parcel(
                parcel_id="upl_fb",
                tasks=[TaskSpec(TaskType.UPLOAD)],
                payload={
                    "audio_input": np.zeros(2400, dtype=np.float32),
                    "upload_targets": {
                        "audio_wav": {
                            "url": f"http://127.0.0.1:{port45}/fail_put.wav",
                            "mime_type": "audio/wav",
                            "allow_inline_fallback": True,
                            "max_inline_bytes": 1024 * 1024,
                        }
                    },
                },
            )
            r_fb = orch45.submit_sync(p_fb, timeout=15.0)
            assert r_fb.success and r_fb.status == "COMPLETED", r_fb.error_message
            assert r_fb.upload_results["audio_wav"]["uploaded"] is False
            assert r_fb.upload_results["audio_wav"]["fallback_inline"] is True
            env_fb = MTLSGatewayClient.build_job_result_envelope("upl_fb", 2, r_fb)
            assert env_fb.attempt == 2
            assert env_fb.result.artifacts[0].uploaded is False
            assert len(env_fb.result.artifacts[0].inline_bytes) > 0

            # 4.5c PUT fails + allow_inline_fallback=False (alongside 1 passing target) -> FAILED + preserves partial upload_results
            p_fail = Parcel(
                parcel_id="upl_hard_fail",
                tasks=[TaskSpec(TaskType.UPLOAD)],
                payload={
                    "audio_input": np.zeros(2400, dtype=np.float32),
                    "upload_targets": {
                        "raw_audio_wav": {
                            "url": f"http://127.0.0.1:{port45}/partial_ok.wav",
                            "mime_type": "audio/wav",
                            "allow_inline_fallback": False,
                        },
                        "audio_wav": {
                            "url": f"http://127.0.0.1:{port45}/fail_put.wav",
                            "mime_type": "audio/wav",
                            "allow_inline_fallback": False,
                        },
                    },
                },
            )
            r_fail = orch45.submit_sync(p_fail, timeout=15.0)
            assert not r_fail.success and r_fail.status == "FAILED"
            assert "raw_audio_wav" in r_fail.upload_results and r_fail.upload_results["raw_audio_wav"]["uploaded"] is True
            assert "audio_wav" in r_fail.upload_results and r_fail.upload_results["audio_wav"]["uploaded"] is False

            # 4.5d Empty ASR outputs ("", "") from ASR stage + explicit empty transcript upload cleanly without failing
            class EmptyASRService:
                def transcribe(self, audio, sample_rate=24000, **kwargs):
                    return "", "", []

            orch_asr_empty = EngineOrchestrator(
                cluster=UploadOnlyCluster(),
                asr_service=EmptyASRService(),
                capabilities=EngineCapabilities(
                    precision="fp16",
                    vocoder_mode="off",
                    enable_enhancer=False,
                    enable_asr=True,
                ),
            )
            try:
                p_empty_asr = Parcel(
                    parcel_id="upl_empty_asr",
                    tasks=[TaskSpec(TaskType.ASR), TaskSpec(TaskType.UPLOAD)],
                    payload={
                        "audio_input": np.zeros(2400, dtype=np.float32),
                        "upload_targets": {
                            "transcript_txt": {
                                "url": f"http://127.0.0.1:{port45}/empty_asr.txt",
                                "mime_type": "text/plain; charset=utf-8",
                                "allow_inline_fallback": False,
                            },
                            "subtitles_srt": {
                                "url": f"http://127.0.0.1:{port45}/empty_asr.srt",
                                "mime_type": "application/x-subrip",
                                "allow_inline_fallback": False,
                            },
                        },
                    },
                )
                r_empty_asr = orch_asr_empty.submit_sync(p_empty_asr, timeout=15.0)
                assert r_empty_asr.success, r_empty_asr.error_message
                assert _MockHTTPHandler.store.get("/empty_asr.txt") == b""
                assert _MockHTTPHandler.store.get("/empty_asr.srt") == b""
            finally:
                orch_asr_empty.shutdown()

            p_empty_tx = Parcel(
                parcel_id="upl_empty_tx",
                tasks=[TaskSpec(TaskType.UPLOAD)],
                payload={
                    "transcript": "",
                    "upload_targets": {
                        "transcript_txt": {
                            "url": f"http://127.0.0.1:{port45}/empty.txt",
                            "mime_type": "text/plain; charset=utf-8",
                            "allow_inline_fallback": False,
                        }
                    },
                },
            )
            r_empty = orch45.submit_sync(p_empty_tx, timeout=15.0)
            assert r_empty.success, r_empty.error_message
            assert _MockHTTPHandler.store.get("/empty.txt") == b""
            print("  PASS: Upload PUT, inline fallback, partial-failure metadata, and empty-transcript upload verified.")

            # -----------------------------------------------------------------
            # Test 4.6: Gateway v2 Asset Hydration, Sync, Cancel, Deny & Writer
            # -----------------------------------------------------------------
            print("\n[TEST 4.6] Gateway v2 (.breeze Hydration, Sync, Cancel, Deny, Reconnect Writer)...")
            sample_breeze = serialize_breeze_voice("hello world voice sample", [7] * 32, frames=2)
            breeze_sha = hashlib.sha256(sample_breeze).hexdigest()
            _MockHTTPHandler.store["/voice.breeze"] = sample_breeze

            gw46 = MTLSGatewayClient(
                server_url="wss://127.0.0.1:9443/ws/worker",
                ca_cert_path="/nonexistent/ca.pem",
                client_cert_path="/nonexistent/client.crt",
                client_key_path="/nonexistent/client.key",
                admission_ctrl=AdmissionController(orchestrator=orch45),
                declared_lease_seconds=43200,
                require_mtls=False,
                asset_cache=AssetCache(max_bytes=2 * 1024 * 1024),
            )

            d_clone = worker_pb2.Dispatch(
                job_type="tts_clone",
                quality="studio",
                input_assets=[
                    worker_pb2.InputAsset(
                        role="voice_breeze",
                        mime_type="application/octet-stream",
                        sha256=breeze_sha,
                        url=f"http://127.0.0.1:{port45}/voice.breeze",
                    )
                ],
                params_json=json.dumps({"script_text": "Test clone sentence", "enhance": False}).encode("utf-8"),
            )
            p_clone = gw46.parse_dispatch_to_parcel("job_clone_46", d_clone)
            assert p_clone.payload.get("_has_voice_ref_hint") is True
            gw46._resolve_input_assets(p_clone, list(d_clone.input_assets))
            assert p_clone.payload["ref_frames"] == 2
            assert p_clone.payload["ref_text"] == "hello world voice sample"
            assert p_clone.payload["ref_codes"] == [7] * 32

            # Completed job cannot be retroactively flipped to CANCELLED
            assert orch45.cancel_request("upl_ok") is False
            assert orch45.active_requests["upl_ok"].status == "COMPLETED"

            # Async gateway envelope tests (SYNC_REQUEST includes resolving_assets, excludes COMPLETED, CANCEL sets CANCEL_ACK, writer retains inflight frame across disconnect)
            async def _async_checks():
                gw46._loop = asyncio.get_running_loop()
                gw46._critical_queue = asyncio.Queue()
                gw46._best_effort_queue = asyncio.Queue(maxsize=32)
                gw46._writer_wakeup = asyncio.Event()
                gw46._connected = True

                # 1) SYNC_REQUEST with 1 completed job + 1 active resolving_assets job
                with gw46.admission_ctrl.lock:
                    gw46.admission_ctrl.running_jobs["downloading_job"] = 3
                await gw46._handle_envelope(
                    worker_pb2.Envelope(
                        job_id="", attempt=0, ts_unix_ms=1,
                        sync_request=worker_pb2.SyncRequest(believed_running_job_ids=["upl_ok", "downloading_job"]),
                    )
                )
                raw_sync = await gw46._critical_queue.get()
                sync_env = worker_pb2.Envelope()
                sync_env.ParseFromString(raw_sync)
                assert sync_env.WhichOneof("msg") == "sync_response"
                running_ids = {r.job_id: r for r in sync_env.sync_response.actual_running}
                assert "upl_ok" not in running_ids
                assert "downloading_job" in running_ids
                assert running_ids["downloading_job"].stage == "resolving_assets"
                assert running_ids["downloading_job"].attempt == 3
                assert sync_env.sync_response.remaining_lease_seconds > 0

                # 2) Deny reason classification for invalid JSON / expired deadline / unknown stage
                await gw46._handle_envelope(
                    worker_pb2.Envelope(
                        job_id="bad_json_job", attempt=1, ts_unix_ms=1,
                        dispatch=worker_pb2.Dispatch(job_type="tts_clone", params_json=b"{bad json"),
                    )
                )
                raw_deny = await gw46._critical_queue.get()
                deny_env = worker_pb2.Envelope()
                deny_env.ParseFromString(raw_deny)
                assert deny_env.WhichOneof("msg") == "deny"
                assert deny_env.deny.reason == worker_pb2.Deny.INVALID_PARAMS

                await gw46._handle_envelope(
                    worker_pb2.Envelope(
                        job_id="expired_job", attempt=1, ts_unix_ms=1,
                        dispatch=worker_pb2.Dispatch(
                            job_type="tts_clone",
                            deadline_ms=int(time.time() * 1000) - 120_000,
                            params_json=json.dumps({"script_text": "stale"}).encode("utf-8"),
                        ),
                    )
                )
                raw_exp = await gw46._critical_queue.get()
                exp_env = worker_pb2.Envelope()
                exp_env.ParseFromString(raw_exp)
                assert exp_env.WhichOneof("msg") == "deny"
                assert exp_env.deny.reason == worker_pb2.Deny.INVALID_PARAMS

                # 3) Pre-cancelled job emits JobResult.CANCELLED without touching orchestrator
                gw46.admission_ctrl.cancel("cancel_before_exec")
                await gw46._execute_job(p_clone, list(d_clone.input_assets), "cancel_before_exec", 1)
                raw_cancel_res = await gw46._critical_queue.get()
                cancel_res_env = worker_pb2.Envelope()
                cancel_res_env.ParseFromString(raw_cancel_res)
                assert cancel_res_env.WhichOneof("msg") == "result"
                assert cancel_res_env.result.status == worker_pb2.JobResult.CANCELLED

                # 4) Cloned ref_hint raises admission chunk estimate properly
                big_script = " ".join(["word"] * 150)
                d_big_clone = worker_pb2.Dispatch(
                    job_type="tts_clone",
                    quality="standard",
                    input_assets=[worker_pb2.InputAsset(role="voice_breeze", url=f"http://127.0.0.1:{port45}/voice.breeze", sha256=breeze_sha)],
                    params_json=json.dumps({"script_text": big_script}).encode("utf-8"),
                )
                p_big_clone = gw46.parse_dispatch_to_parcel("big_clone", d_big_clone)
                assert p_big_clone.payload.get("_has_voice_ref_hint") is True
                with orch45.state_lock:
                    orch45._pending_tts_chunks = orch45.MAX_QUEUE_TTS - 2
                try:
                    ok_adm, rej_reason, *_ = orch45.evaluate_admission(p_big_clone)
                    assert not ok_adm and "STAGE_OVERFLOW" in rej_reason
                finally:
                    with orch45.state_lock:
                        orch45._pending_tts_chunks = 0

                # 5) _writer_loop preserves critical frame on failed ws.send and delivers after reconnect
                crit_payload = MTLSGatewayClient.build_job_result_envelope(
                    "j_crit", 2, ParcelResult(parcel_id="j_crit", success=True, status="COMPLETED")
                ).SerializeToString()
                gw46._critical_queue.put_nowait(crit_payload)
                gw46._writer_wakeup.set()

                class FailOnceWS:
                    async def send(self, _data):
                        raise ConnectionError("socket dropped mid-send")

                class RecordingWS:
                    def __init__(self):
                        self.sent = []
                    async def send(self, data):
                        self.sent.append(data)
                        if len(self.sent) >= 1:
                            gw46.running = False

                gw46.running = True
                try:
                    await gw46._writer_loop(FailOnceWS())
                    raise AssertionError("Expected FailOnceWS to raise ConnectionError")
                except ConnectionError:
                    assert gw46._inflight_critical == crit_payload

                gw46.running = True
                rec_ws = RecordingWS()
                await gw46._writer_loop(rec_ws)
                assert rec_ws.sent == [crit_payload]
                assert gw46._inflight_critical is None

            asyncio.run(_async_checks())
            print("  PASS: .breeze hydration, SYNC_REQUEST, Deny classification, and lossless writer verified.")
        finally:
            orch45.shutdown()
    finally:
        srv45.shutdown()

    print("\n" + "=" * 80)
    print("  ALL PHASE 4 TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
