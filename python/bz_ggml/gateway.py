"""
Production Gateway, Admission Controller, Standalone Telemetry Service & Outbound mTLS.

Connects GPU Compute Nodes outward to the Go Monolith over a single mTLS WebSocket (TLS 1.3):
  1. 100% Binary Protobuf Wire Contract: workerpb.Envelope v2 (single-pass oneof, zero JSON).
  2. Job-Type-Agnostic Parcel Dispatch: Supports explicit stages[], InputAsset (presigned GET URL
     with local SHA-256 LRU cache OR inline bytes), and OutputTarget (presigned PUT URL with
     transient retries + bounded inline_bytes fallback).
  3. Reconnect-Safe Single Writer: Outbound queue bound to the active socket so mid-job
     reconnects never drop terminal JobResult or Ack frames.
  4. Driver-Level GPU Telemetry: Uses torch.cuda.mem_get_info + pynvml for accurate multi-GPU
     VRAM & SM utilization (including GGML C++ CUDA allocations).
  5. Strict Attempt Fencing & Cooperative Cancellation: Echoes exact attempt on every frame.
"""

import asyncio
from collections import deque
import hashlib
import json
import logging
import os
import ssl
import threading
import time
from typing import Optional, List, Dict, Any, Tuple, Callable

from .asset_cache import AssetCache
from .bindings import deserialize_breeze_voice
from .orchestrator import EngineOrchestrator, Parcel, TaskType, TaskSpec, ParcelResult
from .proto import worker_pb2

logger = logging.getLogger("bz_ggml.gateway")


class AdmissionController:
    """
    Deterministic Multi-Stage Admission Controller.
    Delegates capacity checks directly to EngineOrchestrator's decoupled stage queues
    and tracks bounded cancellation history + active job fencing attempts.
    """

    def __init__(
        self,
        orchestrator: EngineOrchestrator,
        max_queue_wait_s: float = 300.0,
        max_queue_depth: Optional[int] = None,
        max_cancel_history: int = 4096,
        **kwargs,
    ):
        self.orchestrator = orchestrator
        self.cluster = getattr(orchestrator, "cluster", None)
        self.max_queue_wait_s = max_queue_wait_s
        self.max_queue_depth = max_queue_depth
        self.lock = threading.RLock()

        # Bounded O(1) cancellation set to prevent unbounded RAM growth over weeks of uptime
        self._cancel_maxlen = max(64, int(max_cancel_history))
        self._cancelled_order: deque = deque()
        self.cancelled_job_ids: set = set()

        self.running_jobs: Dict[str, int] = {}  # job_id -> attempt

    def estimate_parcel_duration(self, parcel: Parcel) -> float:
        return self.orchestrator.estimate_parcel_duration(parcel)

    def evaluate_admission(self, parcel: Parcel) -> Tuple[bool, str, float, float, int]:
        """
        Determines whether a parcel is admitted into the engine.
        Returns: (accepted, reason, estimated_duration_s, eta_s, queue_position)
        """
        with self.lock:
            if parcel.parcel_id in self.cancelled_job_ids:
                return (False, "REJECTED_CANCELLED: Parcel was pre-cancelled.", 0.0, 0.0, 0)
            return self.orchestrator.evaluate_admission(parcel)

    def cancel(self, job_id: str) -> bool:
        """Cancels a job and immediately aborts any queued or active GPU/upload stages."""
        with self.lock:
            if job_id not in self.cancelled_job_ids:
                if len(self._cancelled_order) >= self._cancel_maxlen:
                    evicted = self._cancelled_order.popleft()
                    self.cancelled_job_ids.discard(evicted)
                self._cancelled_order.append(job_id)
                self.cancelled_job_ids.add(job_id)
            return self.orchestrator.cancel_request(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self.lock:
            return job_id in self.cancelled_job_ids

    def submit_sync(
        self,
        parcel: Parcel,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        skip_admission: bool = False,
        timeout: float = 1200.0,
    ) -> Tuple[bool, str, float, float, Optional[ParcelResult]]:
        """
        Synchronous admission and execution entrypoint (used by Gradio and direct callers).
        Returns: (accepted, status_str, duration_s, eta_s, ParcelResult)
        """
        dur_s, eta_s = 0.0, 0.0
        if not skip_admission:
            accepted, reason, dur_s, eta_s, _ = self.evaluate_admission(parcel)
            if not accepted:
                return accepted, reason, dur_s, eta_s, None

        result = self.orchestrator.submit_sync(
            parcel,
            progress_callback=progress_callback,
            timeout=timeout,
            skip_admission=True,
        )
        status_str = result.status if result else "FAILED"
        return bool(result and result.success), status_str, dur_s, eta_s, result


class TelemetrySampler:
    """
    Ultra-Low CPU Usage Hardware Telemetry Sampler (< 0.05% CPU core).
    Queries CUDA driver memory (includes GGML + PyTorch allocations), pynvml SM util,
    kernel CPU/RAM stats, and Orchestrator stage queue snapshots.
    """

    def __init__(self, node_id: str, orchestrator: EngineOrchestrator):
        self.node_id = node_id
        self.orchestrator = orchestrator
        self.cluster = getattr(orchestrator, "cluster", None)

        self._has_torch_cuda = False
        self._gpu_count = 0
        try:
            import torch
            self._torch = torch
            self._has_torch_cuda = torch.cuda.is_available()
            if self._has_torch_cuda:
                self._gpu_count = torch.cuda.device_count()
        except ImportError:
            self._torch = None

        try:
            import psutil
            self._psutil = psutil
            self._psutil.cpu_percent(interval=None)
        except ImportError:
            self._psutil = None

        self._nvml = None
        self._nvml_handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            cnt = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(cnt)]
            if self._gpu_count == 0:
                self._gpu_count = cnt
        except Exception:
            self._nvml = None
            self._nvml_handles = []

    def get_gpu_inventory(self) -> List[Dict[str, Any]]:
        """Returns static GPU inventory [{index, model, vram_total_mb}, ...]."""
        inventory: List[Dict[str, Any]] = []
        count = max(self._gpu_count, len(self._nvml_handles))
        for idx in range(count):
            model = f"GPU-{idx}"
            total_mb = 0
            if self._has_torch_cuda and idx < self._torch.cuda.device_count():
                try:
                    props = self._torch.cuda.get_device_properties(idx)
                    model = str(props.name)
                    total_mb = int(props.total_memory // (1024 * 1024))
                except Exception:
                    pass
            if total_mb == 0 and self._nvml and idx < len(self._nvml_handles):
                try:
                    name = self._nvml.nvmlDeviceGetName(self._nvml_handles[idx])
                    model = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handles[idx])
                    total_mb = int(mem.total // (1024 * 1024))
                except Exception:
                    pass
            inventory.append({"index": idx, "model": model, "vram_total_mb": total_mb})
        return inventory

    def sample_metrics(self) -> Dict[str, Any]:
        """Collects full hardware & queue metrics across all GPUs."""
        t_now_ms = int(time.time() * 1000)

        cpu_pct = 0.0
        ram_used_mb = 0
        ram_total_mb = 0
        if self._psutil:
            try:
                cpu_pct = float(self._psutil.cpu_percent(interval=None))
                vm = self._psutil.virtual_memory()
                ram_used_mb = int(vm.used // (1024 * 1024))
                ram_total_mb = int(vm.total // (1024 * 1024))
            except Exception:
                pass

        coord = getattr(self.cluster, "dynamic_coord", None) if self.cluster is not None else None
        count = max(self._gpu_count, len(self._nvml_handles))
        gpus: List[Dict[str, Any]] = []

        for idx in range(count):
            vram_used_mb = 0
            vram_total_mb = 0
            util_pct = 0.0

            if self._has_torch_cuda and idx < self._torch.cuda.device_count():
                try:
                    free_b, total_b = self._torch.cuda.mem_get_info(idx)
                    vram_used_mb = int((total_b - free_b) // (1024 * 1024))
                    vram_total_mb = int(total_b // (1024 * 1024))
                except Exception:
                    pass

            if self._nvml and idx < len(self._nvml_handles):
                try:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handles[idx])
                    util_pct = float(u.gpu)
                    if vram_total_mb == 0:
                        m = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handles[idx])
                        vram_used_mb = int(m.used // (1024 * 1024))
                        vram_total_mb = int(m.total // (1024 * 1024))
                except Exception:
                    pass

            active_load = int(coord.active_tokens.get(idx, 0)) if coord is not None else 0
            load_ceiling = int(coord.get_admission_ceiling(idx)) if coord is not None else 0

            gpus.append({
                "index": idx,
                "sm_util_pct": util_pct,
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "active_load": active_load,
                "load_ceiling": load_ceiling,
            })

        orch_snapshot = self.orchestrator.get_telemetry_snapshot()

        # Backward-compatible flat GPU 0/1 keys alongside generic gpus[] list
        g0 = gpus[0] if len(gpus) > 0 else {"vram_used_mb": 0, "vram_total_mb": 0, "sm_util_pct": 0.0, "active_load": 0, "load_ceiling": 0}
        g1 = gpus[1] if len(gpus) > 1 else {"vram_used_mb": 0, "vram_total_mb": 0, "sm_util_pct": 0.0, "active_load": 0, "load_ceiling": 0}

        return {
            "node_id": self.node_id,
            "timestamp_ms": t_now_ms,
            "cpu_percent": cpu_pct,
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "ram_used_gb": round(ram_used_mb / 1024.0, 2),
            "ram_total_gb": round(ram_total_mb / 1024.0, 2),
            "gpus": gpus,
            "gpu0_vram_used_mb": g0["vram_used_mb"],
            "gpu0_vram_total_mb": g0["vram_total_mb"],
            "gpu0_util_percent": g0["sm_util_pct"],
            "gpu0_active_tokens": g0["active_load"],
            "gpu0_token_ceiling": g0["load_ceiling"],
            "gpu1_vram_used_mb": g1["vram_used_mb"],
            "gpu1_vram_total_mb": g1["vram_total_mb"],
            "gpu1_util_percent": g1["sm_util_pct"],
            "gpu1_active_tokens": g1["active_load"],
            "gpu1_token_ceiling": g1["load_ceiling"],
            "queue_depths": orch_snapshot.get("queue_depths", {}),
            "active_requests": orch_snapshot.get("active_requests", []),
        }


class TelemetryService:
    """
    Standalone local Telemetry Service (used by Gradio UI or local diagnostics).
    Runs in an independent background daemon thread on a 1.0-second timer.
    """

    def __init__(self, sampler: TelemetrySampler, broadcast_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.sampler = sampler
        self.broadcast_callback = broadcast_callback
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TelemetryDaemon")
        self._thread.start()

    def stop(self):
        self.running = False

    def _run_loop(self):
        while self.running:
            t0 = time.time()
            try:
                metrics = self.sampler.sample_metrics()
                if self.broadcast_callback:
                    self.broadcast_callback(metrics)
            except Exception as e:
                logger.debug(f"[TelemetryService] Sampling exception: {e}")

            elapsed = time.time() - t0
            time.sleep(max(0.1, 1.0 - elapsed))


def _classify_deny_reason(reason_text: str, default_reason: Optional[int] = None) -> int:
    upper = (reason_text or "").upper()
    if "STAGE_OVERFLOW" in upper or "SATURATED" in upper:
        return worker_pb2.Deny.STAGE_SATURATED
    if "VRAM" in upper or "OOM" in upper:
        return worker_pb2.Deny.VRAM_PRESSURE
    if "DRAIN" in upper:
        return worker_pb2.Deny.DRAINING
    if (
        "DISABLED" in upper
        or "UNSUPPORTED" in upper
        or "NOT A VALID TASKTYPE" in upper
        or "NOT SUPPORTED" in upper
    ):
        return worker_pb2.Deny.UNSUPPORTED
    if (
        "REJECTED_CANCELLED" in upper
        or "PRE-CANCELLED" in upper
        or "EXPIRED" in upper
        or "DEADLINE" in upper
        or "INVALID" in upper
        or "MISSING" in upper
        or "EMPTY" in upper
        or "MUST BE" in upper
        or "MUST STRICTLY" in upper
        or "REQUIRES" in upper
        or "JSONDECODE" in upper
        or "EXPECTING VALUE" in upper
    ):
        return worker_pb2.Deny.INVALID_PARAMS
    if default_reason is not None:
        return default_reason
    return worker_pb2.Deny.QUEUE_FULL


class MTLSGatewayClient:
    """
    Outbound Mutual TLS (mTLS) Gateway Client (100% Binary Protobuf workerpb v2).

    Connects outward from GPU worker to the Go Monolith over a single TLS 1.3 WebSocket.
    All writes pass through a reconnect-safe outbound queue drained by a single writer task
    attached to the currently live socket, ensuring terminal JobResult frames survive drops.
    """

    def __init__(
        self,
        server_url: str,
        ca_cert_path: str,
        client_cert_path: str,
        client_key_path: str,
        admission_ctrl: AdmissionController,
        node_id: Optional[str] = None,
        instance_type: str = "dual_t4",
        telemetry_interval_s: float = 1.0,
        declared_lease_seconds: int = 0,
        global_max_concurrent: int = 16,
        require_mtls: bool = True,
        asset_cache: Optional[AssetCache] = None,
    ):
        self.server_url = server_url
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.admission_ctrl = admission_ctrl
        self.orchestrator = admission_ctrl.orchestrator
        self.node_id = node_id or f"node_{os.uname().nodename}_{int(time.time())}"
        self.instance_type = instance_type
        self.telemetry_interval_s = max(0.5, float(telemetry_interval_s))
        self.declared_lease_seconds = max(0, int(declared_lease_seconds))
        self.global_max_concurrent = max(1, int(global_max_concurrent))
        self.require_mtls = require_mtls
        self.worker_id: Optional[str] = None

        self._started_at = time.time()
        self.sampler = TelemetrySampler(self.node_id, self.orchestrator)
        self.asset_cache = asset_cache or AssetCache()
        self.ssl_context = self._create_ssl_context()

        self.running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        self._inflight_critical: Optional[bytes] = None
        self._writer_wakeup: Optional[asyncio.Event] = None
        self._critical_queue: Optional[asyncio.Queue] = None
        self._best_effort_queue: Optional[asyncio.Queue] = None

    def _remaining_lease_seconds(self) -> int:
        if self.declared_lease_seconds <= 0:
            return 0
        elapsed = int(time.time() - self._started_at)
        return max(0, self.declared_lease_seconds - elapsed)

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Configures Mutual TLS 1.3 Context with Client and Server Certificates."""
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if hasattr(ssl, "TLSVersion"):
            ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        if os.path.exists(self.ca_cert_path):
            ctx.load_verify_locations(cafile=self.ca_cert_path)
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.check_hostname = False
        elif self.require_mtls:
            raise FileNotFoundError(
                f"mTLS CA certificate not found at '{self.ca_cert_path}'. "
                "Pass require_mtls=False only for local dev."
            )
        else:
            logger.warning(f"CA certificate '{self.ca_cert_path}' not found; using unverified SSL (require_mtls=False).")
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        if os.path.exists(self.client_cert_path) and os.path.exists(self.client_key_path):
            ctx.load_cert_chain(certfile=self.client_cert_path, keyfile=self.client_key_path)
            logger.info("Loaded client certificate and private key for mTLS.")
        elif self.require_mtls:
            raise FileNotFoundError(
                f"mTLS client cert/key not found ('{self.client_cert_path}', '{self.client_key_path}')."
            )

        return ctx

    def _enqueue_envelope(self, env: worker_pb2.Envelope, *, critical: bool = True) -> None:
        """Thread-safe enqueue of a serialized Protobuf Envelope to the active writer loop."""
        payload = env.SerializeToString()
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _put():
            woke = False
            if critical:
                if self._critical_queue is not None:
                    self._critical_queue.put_nowait(payload)
                    woke = True
            else:
                if self._connected and self._best_effort_queue is not None:
                    if self._best_effort_queue.full():
                        try:
                            self._best_effort_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._best_effort_queue.put_nowait(payload)
                        woke = True
                    except asyncio.QueueFull:
                        pass
            if woke and self._writer_wakeup is not None:
                self._writer_wakeup.set()

        loop.call_soon_threadsafe(_put)

    def _clear_best_effort_queue(self) -> None:
        if self._best_effort_queue is None:
            return
        while True:
            try:
                self._best_effort_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def run_forever(self):
        """Persistent outbound mTLS WebSocket loop with exponential backoff reconnects."""
        import websockets

        self.running = True
        self._loop = asyncio.get_running_loop()
        if self._critical_queue is None:
            self._critical_queue = asyncio.Queue()
        if self._best_effort_queue is None:
            self._best_effort_queue = asyncio.Queue(maxsize=128)
        if self._writer_wakeup is None:
            self._writer_wakeup = asyncio.Event()

        backoff = 1.0

        while self.running:
            try:
                logger.info(f"[mTLS Gateway] Connecting outbound to {self.server_url} (Node: {self.node_id})...")
                async with websockets.connect(
                    self.server_url,
                    ssl=self.ssl_context,
                    max_size=64 * 1024 * 1024,
                    ping_interval=5,
                    ping_timeout=10,
                ) as ws:
                    logger.info("[mTLS Gateway] Secure bi-directional mTLS stream established!")
                    backoff = 1.0
                    self._connected = True

                    # 1. Immediately send REGISTER envelope first before draining pending results
                    await ws.send(self.build_register_envelope().SerializeToString())
                    logger.info("[mTLS Gateway] Outbound REGISTER envelope sent to platform.")

                    # 2. Run single writer, telemetry ticker, and message receiver concurrently
                    writer_task = asyncio.create_task(self._writer_loop(ws))
                    telemetry_task = asyncio.create_task(self._telemetry_loop())
                    receiver_task = asyncio.create_task(self._receiver_loop(ws))
                    tasks = [writer_task, telemetry_task, receiver_task]
                    try:
                        done, pending = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        for task in done:
                            if not task.cancelled():
                                exc = task.exception()
                                if exc is not None:
                                    raise exc
                    finally:
                        self._connected = False
                        self._clear_best_effort_queue()
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                self._connected = False
                self._clear_best_effort_queue()
                if not self.running:
                    break
                logger.warning(f"[mTLS Gateway] Connection error: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.5)

    async def _writer_loop(self, ws):
        """Lossless single WebSocket writer; retains inflight critical frames across socket drops."""
        assert self._critical_queue is not None and self._best_effort_queue is not None
        while self.running:
            if self._inflight_critical is not None:
                await ws.send(self._inflight_critical)
                self._inflight_critical = None
                continue

            if not self._critical_queue.empty():
                self._inflight_critical = self._critical_queue.get_nowait()
                await ws.send(self._inflight_critical)
                self._inflight_critical = None
                continue

            if not self._best_effort_queue.empty():
                frame = self._best_effort_queue.get_nowait()
                await ws.send(frame)
                continue

            if self._writer_wakeup is None:
                await asyncio.sleep(0.01)
                continue

            self._writer_wakeup.clear()
            if (
                self._inflight_critical is not None
                or not self._critical_queue.empty()
                or not self._best_effort_queue.empty()
            ):
                continue
            await self._writer_wakeup.wait()

    def build_register_envelope(self) -> worker_pb2.Envelope:
        """Builds dynamic Register envelope reflecting actual hardware and enabled capabilities."""
        gpu_infos = [
            worker_pb2.GPUInfo(
                index=g["index"],
                model=g["model"],
                vram_total_mb=g["vram_total_mb"],
            )
            for g in self.sampler.get_gpu_inventory()
        ]

        caps_cfg = getattr(self.orchestrator, "capabilities", None)
        enhancer_enabled = bool(getattr(caps_cfg, "enable_enhancer", True))
        asr_enabled = bool(getattr(caps_cfg, "enable_asr", True))

        tts_tiers = ["standard", "studio"] if enhancer_enabled else ["standard"]
        cap_types = [
            worker_pb2.Capability(
                name="tts_clone",
                quality_tiers=tts_tiers,
                max_concurrent_jobs=self.global_max_concurrent,
                throughput_unit_per_sec=18.0,
            ),
            worker_pb2.Capability(
                name="tts_clone_edit",
                quality_tiers=tts_tiers,
                max_concurrent_jobs=self.global_max_concurrent,
                throughput_unit_per_sec=18.0,
            ),
            worker_pb2.Capability(
                name="tts_voice_description",
                quality_tiers=tts_tiers,
                max_concurrent_jobs=self.global_max_concurrent,
                throughput_unit_per_sec=18.0,
            ),
            worker_pb2.Capability(
                name="voice_registration",
                quality_tiers=["standard"],
                max_concurrent_jobs=4,
                throughput_unit_per_sec=5.0,
            ),
        ]
        if enhancer_enabled:
            cap_types.append(
                worker_pb2.Capability(
                    name="audio_enhance",
                    quality_tiers=["standard"],
                    max_concurrent_jobs=4,
                    throughput_unit_per_sec=1.2,
                )
            )
        if asr_enabled:
            cap_types.append(
                worker_pb2.Capability(
                    name="speech_to_text",
                    quality_tiers=["turbo"],
                    max_concurrent_jobs=4,
                    throughput_unit_per_sec=30.0,
                )
            )

        stage_limits = {
            "tts": int(getattr(EngineOrchestrator, "MAX_QUEUE_TTS", 150)),
            "enhance": int(getattr(EngineOrchestrator, "MAX_QUEUE_ENHANCE", 8)) if enhancer_enabled else 0,
            "asr": int(getattr(EngineOrchestrator, "MAX_QUEUE_ASR", 20)) if asr_enabled else 0,
            "voice_reg": int(getattr(EngineOrchestrator, "MAX_QUEUE_VOICE_REG", 15)),
            "master": int(getattr(EngineOrchestrator, "MAX_QUEUE_MASTER", 20)),
            "upload": int(getattr(EngineOrchestrator, "MAX_QUEUE_UPLOAD", 100)),
        }

        reg = worker_pb2.Register(
            node_id=self.node_id,
            instance_type=self.instance_type,
            gpus=gpu_infos,
            capabilities=cap_types,
            global_max_concurrent=self.global_max_concurrent,
            stage_queue_limits=stage_limits,
            declared_lease_seconds=self._remaining_lease_seconds(),
        )
        return worker_pb2.Envelope(
            job_id="",
            attempt=0,
            ts_unix_ms=int(time.time() * 1000),
            register=reg,
        )

    def _iter_resolving_jobs(self, metrics: Dict[str, Any]) -> List[Tuple[str, int]]:
        """Returns (job_id, attempt) for dispatched jobs still downloading InputAssets."""
        known_ids = {r.get("request_id") for r in metrics.get("active_requests", [])}
        resolving: List[Tuple[str, int]] = []
        with self.admission_ctrl.lock:
            for rid, att in self.admission_ctrl.running_jobs.items():
                if rid not in known_ids and rid not in self.admission_ctrl.cancelled_job_ids:
                    resolving.append((rid, int(att)))
        return resolving

    def build_telemetry_envelope(self) -> worker_pb2.Envelope:
        """Builds a binary Protobuf Telemetry envelope from live driver & orchestrator metrics."""
        metrics = self.sampler.sample_metrics()
        gpu_msgs = [
            worker_pb2.GPUTelemetry(
                index=int(g["index"]),
                sm_util_pct=float(g["sm_util_pct"]),
                vram_used_mb=int(g["vram_used_mb"]),
                vram_total_mb=int(g["vram_total_mb"]),
                active_load=int(g["active_load"]),
                load_ceiling=int(g["load_ceiling"]),
            )
            for g in metrics.get("gpus", [])
        ]

        active_count = sum(
            1 for r in metrics.get("active_requests", [])
            if r.get("status") in ("QUEUED", "RUNNING")
        ) + len(self._iter_resolving_jobs(metrics))
        stage_depths = {str(k): int(v) for k, v in metrics.get("queue_depths", {}).items()}

        tel = worker_pb2.Telemetry(
            cpu_pct=float(metrics.get("cpu_percent", 0.0)),
            ram_used_mb=int(metrics.get("ram_used_mb", 0)),
            ram_total_mb=int(metrics.get("ram_total_mb", 0)),
            gpus=gpu_msgs,
            stage_queue_depths=stage_depths,
            active_jobs=active_count,
            remaining_lease_seconds=self._remaining_lease_seconds(),
        )
        return worker_pb2.Envelope(
            job_id="",
            attempt=0,
            ts_unix_ms=int(metrics.get("timestamp_ms", int(time.time() * 1000))),
            telemetry=tel,
        )

    async def _telemetry_loop(self):
        """Enqueues binary Protobuf Telemetry envelopes on the configured interval."""
        while self.running:
            try:
                env = self.build_telemetry_envelope()
                self._enqueue_envelope(env, critical=False)
                await asyncio.sleep(self.telemetry_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[mTLS Gateway] Telemetry loop exception: {e}")
                await asyncio.sleep(self.telemetry_interval_s)

    async def _receiver_loop(self, ws):
        """Receives binary workerpb.Envelope frames from the Go Monolith."""
        while self.running:
            raw_msg = await ws.recv()
            if not isinstance(raw_msg, (bytes, bytearray)):
                logger.warning("[mTLS Gateway] Ignoring unexpected non-binary frame from server.")
                continue

            env = worker_pb2.Envelope()
            try:
                env.ParseFromString(raw_msg)
            except Exception as e:
                logger.warning(f"[mTLS Gateway] Failed to decode Protobuf Envelope: {e}")
                continue

            await self._handle_envelope(env)

    async def _handle_envelope(self, env: worker_pb2.Envelope):
        """Dispatches incoming v2 oneof Envelope messages."""
        which = env.WhichOneof("msg")

        if which == "register_ack":
            ack = env.register_ack
            if not ack.accepted:
                logger.error(f"[mTLS Gateway] Registration rejected by server: {ack.rejection_reason}")
                self.running = False
                return
            self.worker_id = ack.worker_id
            if ack.telemetry_interval_ms >= 500:
                self.telemetry_interval_s = ack.telemetry_interval_ms / 1000.0
            logger.info(
                f"[mTLS Gateway] Registered worker_id={self.worker_id} "
                f"(telemetry_interval={self.telemetry_interval_s:.2f}s)"
            )

        elif which == "dispatch":
            dispatch = env.dispatch
            now_ms = int(time.time() * 1000)
            if dispatch.deadline_ms > 0 and (now_ms - int(dispatch.deadline_ms)) > 60_000:
                self._enqueue_envelope(
                    worker_pb2.Envelope(
                        job_id=env.job_id,
                        attempt=env.attempt,
                        ts_unix_ms=now_ms,
                        deny=worker_pb2.Deny(
                            reason=worker_pb2.Deny.INVALID_PARAMS,
                            retry_after_ms=0,
                            detail="Dispatch deadline already expired.",
                        ),
                    ),
                    critical=True,
                )
                return

            try:
                parcel = self.parse_dispatch_to_parcel(env.job_id, dispatch)
                self.orchestrator.validate_parcel(parcel)
            except Exception as e:
                deny = worker_pb2.Deny(
                    reason=_classify_deny_reason(str(e), default_reason=worker_pb2.Deny.INVALID_PARAMS),
                    retry_after_ms=0,
                    detail=str(e),
                )
                self._enqueue_envelope(
                    worker_pb2.Envelope(
                        job_id=env.job_id,
                        attempt=env.attempt,
                        ts_unix_ms=int(time.time() * 1000),
                        deny=deny,
                    ),
                    critical=True,
                )
                return

            with self.admission_ctrl.lock:
                prev_attempt = self.admission_ctrl.running_jobs.get(env.job_id)
                if prev_attempt is not None and env.attempt > prev_attempt:
                    self.orchestrator.cancel_request(env.job_id)

            accepted, reason, _, eta_s, pos = self.admission_ctrl.evaluate_admission(parcel)
            if accepted:
                with self.admission_ctrl.lock:
                    self.admission_ctrl.running_jobs[env.job_id] = env.attempt

                self._enqueue_envelope(
                    worker_pb2.Envelope(
                        job_id=env.job_id,
                        attempt=env.attempt,
                        ts_unix_ms=int(time.time() * 1000),
                        ack=worker_pb2.Ack(eta_seconds=eta_s, queue_position=max(0, int(pos))),
                    ),
                    critical=True,
                )
                asyncio.create_task(
                    self._execute_job(
                        parcel,
                        list(dispatch.input_assets),
                        env.job_id,
                        env.attempt,
                        deadline_ms=int(dispatch.deadline_ms),
                    )
                )
            else:
                deny_code = _classify_deny_reason(reason)
                retry_ms = 0 if deny_code in (worker_pb2.Deny.INVALID_PARAMS, worker_pb2.Deny.UNSUPPORTED) else 2000
                deny = worker_pb2.Deny(
                    reason=deny_code,
                    retry_after_ms=retry_ms,
                    detail=reason,
                )
                self._enqueue_envelope(
                    worker_pb2.Envelope(
                        job_id=env.job_id,
                        attempt=env.attempt,
                        ts_unix_ms=int(time.time() * 1000),
                        deny=deny,
                    ),
                    critical=True,
                )

        elif which == "cancel":
            found = self.admission_ctrl.cancel(env.job_id)
            self._enqueue_envelope(
                worker_pb2.Envelope(
                    job_id=env.job_id,
                    attempt=env.attempt,
                    ts_unix_ms=int(time.time() * 1000),
                    cancel_ack=worker_pb2.CancelAck(found=found),
                ),
                critical=True,
            )

        elif which == "sync_request":
            metrics = self.sampler.sample_metrics()
            running_msgs = []
            with self.admission_ctrl.lock:
                for r in metrics.get("active_requests", []):
                    if r.get("status") not in ("QUEUED", "RUNNING"):
                        continue
                    rid = r["request_id"]
                    running_msgs.append(
                        worker_pb2.SyncResponse.RunningJob(
                            job_id=rid,
                            attempt=int(self.admission_ctrl.running_jobs.get(rid, 1)),
                            stage=str(r.get("current_stage", "")),
                            progress_pct=round(float(r.get("overall_progress", 0.0)) * 100.0, 2),
                        )
                    )
                for rid, att in self._iter_resolving_jobs(metrics):
                    running_msgs.append(
                        worker_pb2.SyncResponse.RunningJob(
                            job_id=rid,
                            attempt=att,
                            stage="resolving_assets",
                            progress_pct=0.0,
                        )
                    )
            free_cap = max(0, self.global_max_concurrent - len(running_msgs))
            sync_resp = worker_pb2.SyncResponse(
                actual_running=running_msgs,
                free_capacity=free_cap,
                remaining_lease_seconds=self._remaining_lease_seconds(),
            )
            self._enqueue_envelope(
                worker_pb2.Envelope(
                    job_id="",
                    attempt=0,
                    ts_unix_ms=int(time.time() * 1000),
                    sync_response=sync_resp,
                ),
                critical=True,
            )

    def _resolve_input_assets(self, parcel: Parcel, input_assets: List[worker_pb2.InputAsset]) -> None:
        """
        Resolves Dispatch.input_assets[] (presigned GET URL via local SHA-256 LRU cache
        or inline_bytes fallback) into parcel.payload before execution.
        """
        if not input_assets:
            return

        resolved_map: Dict[str, bytes] = parcel.payload.setdefault("resolved_assets", {})
        for asset in input_assets:
            src_kind = asset.WhichOneof("source")
            raw_bytes = self.asset_cache.resolve_bytes(
                url=asset.url if src_kind == "url" else "",
                inline_bytes=asset.inline_bytes if src_kind == "inline_bytes" else b"",
                expected_sha256=asset.sha256,
            )
            role = (asset.role or "").strip().lower()
            resolved_map[role] = raw_bytes

            if role == "voice_breeze":
                voice_data = deserialize_breeze_voice(raw_bytes)
                parcel.payload["ref_codes"] = voice_data["codes"]
                parcel.payload["ref_frames"] = int(voice_data.get("frames", voice_data.get("n_frames", 0)))
                tx = voice_data.get("text") or voice_data.get("transcript")
                if tx and not parcel.payload.get("ref_text"):
                    parcel.payload["ref_text"] = tx
            elif role in ("audio_input", "reference_audio", "init_audio"):
                parcel.payload["audio_input"] = raw_bytes

    async def _execute_job(
        self,
        parcel: Parcel,
        input_assets: List[worker_pb2.InputAsset],
        job_id: str,
        attempt: int,
        deadline_ms: int = 0,
    ):
        """Resolves assets, runs parcel in executor thread, streams throttled Progress, and emits JobResult."""
        loop = asyncio.get_running_loop()
        last_progress_state = {"stage": "", "ts": 0.0}

        def progress_cb(stage: str, pct: float, msg: str):
            now = time.time()
            stage_changed = stage != last_progress_state["stage"]
            if not stage_changed and (now - last_progress_state["ts"] < 1.0) and pct < 1.0:
                return
            last_progress_state["stage"] = stage
            last_progress_state["ts"] = now

            prog_env = worker_pb2.Envelope(
                job_id=job_id,
                attempt=attempt,
                ts_unix_ms=int(now * 1000),
                progress=worker_pb2.Progress(
                    stage=str(stage),
                    progress_pct=round(max(0.0, min(100.0, float(pct) * 100.0)), 2),
                    eta_remaining_s=0.0,
                    message=str(msg or ""),
                ),
            )
            self._enqueue_envelope(prog_env, critical=False)

        try:
            def run_blocking():
                if self.admission_ctrl.is_cancelled(job_id):
                    return True, "CANCELLED", 0.0, 0.0, ParcelResult(
                        parcel_id=job_id,
                        success=False,
                        status="CANCELLED",
                        error_message="Request cancelled before execution.",
                    )
                self._resolve_input_assets(parcel, input_assets)
                if self.admission_ctrl.is_cancelled(job_id):
                    return True, "CANCELLED", 0.0, 0.0, ParcelResult(
                        parcel_id=job_id,
                        success=False,
                        status="CANCELLED",
                        error_message="Request cancelled during asset resolution.",
                    )
                timeout_s = 1200.0
                if deadline_ms > 0:
                    remaining_s = (deadline_ms / 1000.0) - time.time()
                    timeout_s = max(5.0, min(1200.0, remaining_s))
                return self.admission_ctrl.submit_sync(
                    parcel,
                    progress_callback=progress_cb,
                    skip_admission=True,
                    timeout=timeout_s,
                )

            _, _, _, _, result = await loop.run_in_executor(None, run_blocking)
            result_env = self.build_job_result_envelope(job_id, attempt, result)
            self._enqueue_envelope(result_env, critical=True)
        except Exception as e:
            err_res = worker_pb2.JobResult(
                status=worker_pb2.JobResult.FAILED,
                error_code="WORKER_EXECUTION_ERROR",
                error_message=str(e),
            )
            self._enqueue_envelope(
                worker_pb2.Envelope(
                    job_id=job_id,
                    attempt=attempt,
                    ts_unix_ms=int(time.time() * 1000),
                    result=err_res,
                ),
                critical=True,
            )
        finally:
            with self.admission_ctrl.lock:
                if self.admission_ctrl.running_jobs.get(job_id) == attempt:
                    self.admission_ctrl.running_jobs.pop(job_id, None)

    @staticmethod
    def build_job_result_envelope(
        job_id: str,
        attempt: int,
        result: Optional[ParcelResult],
    ) -> worker_pb2.Envelope:
        """Converts internal ParcelResult into a generic workerpb.JobResult envelope."""
        if result is None:
            jr = worker_pb2.JobResult(
                status=worker_pb2.JobResult.FAILED,
                error_code="EMPTY_RESULT",
                error_message="Job execution returned no result.",
            )
        else:
            status_str = (result.status or ("COMPLETED" if result.success else "FAILED")).upper()
            if status_str == "CANCELLED":
                pb_status = worker_pb2.JobResult.CANCELLED
            elif result.success and status_str == "COMPLETED":
                pb_status = worker_pb2.JobResult.COMPLETED
            else:
                pb_status = worker_pb2.JobResult.FAILED

            artifacts_pb: List[worker_pb2.ArtifactResult] = []
            handled_keys = set()

            for art_key, info in (result.upload_results or {}).items():
                handled_keys.add(art_key)
                uploaded = bool(info.get("uploaded", False))
                inline_data = b"" if uploaded else (result.artifact_bytes.get(art_key) or b"")
                size_bytes = int(info.get("size_bytes", len(inline_data)))
                sha256_hex = str(info.get("sha256", ""))
                if not sha256_hex and inline_data:
                    sha256_hex = hashlib.sha256(inline_data).hexdigest()

                meta: Dict[str, str] = {}
                if result.duration_s > 0 and "audio" in art_key:
                    meta["duration_s"] = f"{result.duration_s:.2f}"
                    meta["sample_rate"] = str(result.sample_rate)

                artifacts_pb.append(
                    worker_pb2.ArtifactResult(
                        key=art_key,
                        uploaded=uploaded,
                        size_bytes=size_bytes,
                        sha256=sha256_hex,
                        mime_type=str(info.get("mime_type", "application/octet-stream")),
                        inline_bytes=inline_data,
                        meta=meta,
                    )
                )

            # Include any purely inline artifacts (when no UPLOAD stage ran)
            for art_key, raw_bytes in (result.artifact_bytes or {}).items():
                if art_key in handled_keys or not raw_bytes:
                    continue
                artifacts_pb.append(
                    worker_pb2.ArtifactResult(
                        key=art_key,
                        uploaded=False,
                        size_bytes=len(raw_bytes),
                        sha256=hashlib.sha256(raw_bytes).hexdigest(),
                        mime_type="application/octet-stream",
                        inline_bytes=raw_bytes,
                    )
                )

            meta_payload: Dict[str, Any] = {
                "duration_s": result.duration_s,
                "sample_rate": result.sample_rate,
                "chunks_count": result.chunks_count,
            }
            if result.transcript:
                meta_payload["transcript"] = result.transcript

            jr = worker_pb2.JobResult(
                status=pb_status,
                error_code="" if pb_status == worker_pb2.JobResult.COMPLETED else status_str,
                error_message="" if pb_status == worker_pb2.JobResult.COMPLETED else (result.error_message or ""),
                compute_units_used=float(result.duration_s or 0.0),
                artifacts=artifacts_pb,
                stage_timings={str(k): float(v) for k, v in (result.stage_timings or {}).items()},
                metadata_json=json.dumps(meta_payload).encode("utf-8"),
            )

        return worker_pb2.Envelope(
            job_id=job_id,
            attempt=attempt,
            ts_unix_ms=int(time.time() * 1000),
            result=jr,
        )

    _STAGE_ALIASES: Dict[str, TaskType] = {
        "tts": TaskType.TTS,
        "tts_clone": TaskType.TTS,
        "tts_clone_edit": TaskType.TTS,
        "tts_voice_description": TaskType.TTS,
        "stitch": TaskType.STITCH,
        "enhance": TaskType.ENHANCE,
        "audio_enhance": TaskType.ENHANCE,
        "master": TaskType.MASTER,
        "asr": TaskType.ASR,
        "speech_to_text": TaskType.ASR,
        "voice_register": TaskType.VOICE_REGISTER,
        "voice_reg": TaskType.VOICE_REGISTER,
        "voice_registration": TaskType.VOICE_REGISTER,
        "register_voice": TaskType.VOICE_REGISTER,
        "voice_convert": TaskType.VOICE_CONVERT,
        "upload": TaskType.UPLOAD,
    }

    @classmethod
    def _coerce_task_type(cls, stage_name: str) -> TaskType:
        clean = (stage_name or "").strip().lower()
        if clean in cls._STAGE_ALIASES:
            return cls._STAGE_ALIASES[clean]
        return TaskType(clean)

    @classmethod
    def parse_dispatch_to_parcel(cls, job_id: str, dispatch: worker_pb2.Dispatch) -> Parcel:
        """Translates a generic workerpb.Dispatch into an internal EngineOrchestrator Parcel."""
        params_dict: Dict[str, Any] = {}
        if dispatch.params_json:
            parsed = json.loads(dispatch.params_json.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("Invalid Dispatch.params_json: expected JSON object.")
            params_dict = parsed

        j_type = (dispatch.job_type or "").lower()
        quality = (dispatch.quality or "").lower()
        tasks: List[TaskSpec] = []

        if len(dispatch.stages) > 0:
            # Explicit generic pipeline stages provided by Go ServicePlugin
            for st in dispatch.stages:
                st_params = json.loads(st.params_json.decode("utf-8")) if st.params_json else {}
                if not isinstance(st_params, dict):
                    raise ValueError(f"Invalid StageSpec.params_json for '{st.stage_type}': expected JSON object.")
                tasks.append(TaskSpec(cls._coerce_task_type(st.stage_type), st_params))

            # Preserve Option A 30 dB SNR enhancer auto-gate on voice registration pipelines unless explicitly overridden
            is_vreg_pipeline = (
                "voice_reg" in j_type
                or "register_voice" in j_type
                or "voice_registration" in j_type
                or any(t.task_type == TaskType.VOICE_REGISTER for t in tasks)
            )
            if is_vreg_pipeline:
                for t in tasks:
                    if t.task_type == TaskType.ENHANCE and "auto_gate" not in t.params:
                        t.params["auto_gate"] = True
                        t.params.setdefault("min_snr_db", 30.0)
                        t.params.setdefault("chunk_seconds", 10.0)
        elif "tasks" in params_dict and isinstance(params_dict["tasks"], list):
            for t in params_dict["tasks"]:
                tasks.append(TaskSpec(cls._coerce_task_type(t.get("task_type", "tts")), t.get("params", {})))
        elif "enhance" in j_type:
            tasks.append(TaskSpec(TaskType.ENHANCE, params_dict))
        elif "asr" in j_type or "transcribe" in j_type or "translate" in j_type or "speech_to_text" in j_type:
            tasks.append(TaskSpec(TaskType.ASR, params_dict))
        elif "voice_reg" in j_type or "register_voice" in j_type or "voice_registration" in j_type:
            v_enh = params_dict.get("enhance", "auto")
            if v_enh is True:
                tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}))
            elif isinstance(v_enh, str) and v_enh.strip().lower() == "auto":
                tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0, "auto_gate": True, "min_snr_db": 30.0}))
            if params_dict.get("master", True):
                tasks.append(TaskSpec(TaskType.MASTER, {"preset": params_dict.get("preset", "gentle_loudnorm")}))
            explicit_tx = params_dict.get("transcript")
            if not (isinstance(explicit_tx, str) and explicit_tx.strip()):
                tasks.append(TaskSpec(TaskType.ASR, {"output_format": "txt"}))
            tasks.append(TaskSpec(TaskType.VOICE_REGISTER, params_dict))
        elif params_dict.get("return_tokens_only"):
            tasks.append(TaskSpec(TaskType.TTS, params_dict))
        else:
            tasks.append(TaskSpec(TaskType.TTS, params_dict))
            if params_dict.get("enhance", "studio" in quality):
                tasks.append(TaskSpec(TaskType.ENHANCE, {"chunk_seconds": 10.0}))
            if params_dict.get("master", "studio" in quality):
                tasks.append(TaskSpec(TaskType.MASTER, {"preset": params_dict.get("preset", "optionA_rich_bass")}))
            if params_dict.get("srt", False) or params_dict.get("asr", False):
                tasks.append(TaskSpec(TaskType.ASR, params_dict.get("asr_params", {"output_format": "all"})))

        # Flag cloned voice references early so admission estimates 45w cloned chunk counts accurately
        if any((a.role or "").strip().lower() == "voice_breeze" for a in getattr(dispatch, "input_assets", [])) or (
            params_dict.get("ref_codes") and params_dict.get("ref_frames")
        ):
            params_dict["_has_voice_ref_hint"] = True

        # Map typed OutputTarget list into orchestrator upload_targets specs
        upload_targets: Dict[str, Any] = {}
        if len(dispatch.output_targets) > 0:
            for tgt in dispatch.output_targets:
                upload_targets[tgt.key] = {
                    "url": tgt.upload_url,
                    "mime_type": tgt.mime_type,
                    "allow_inline_fallback": bool(tgt.allow_inline_fallback),
                    "max_inline_bytes": int(tgt.max_inline_bytes),
                }
        elif params_dict.get("upload_targets"):
            upload_targets = params_dict["upload_targets"]

        if upload_targets and not any(t.task_type == TaskType.UPLOAD for t in tasks):
            tasks.append(TaskSpec(TaskType.UPLOAD, {"targets": upload_targets}))
        if upload_targets:
            params_dict["upload_targets"] = upload_targets

        return Parcel(
            parcel_id=job_id,
            tasks=tasks,
            payload=params_dict,
            return_artifacts=params_dict.get("return_artifacts"),
            upload_targets=upload_targets or None,
        )
