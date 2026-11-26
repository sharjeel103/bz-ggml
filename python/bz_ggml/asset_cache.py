"""
Bounded Local Input Asset Cache (Presigned GET URL + Inline Bytes Fallback).

Resolves Dispatch.input_assets[] for any modality:
  - Inline bytes: returned directly with optional SHA-256 verification.
  - Presigned GET URL: served from bounded local disk LRU cache keyed by SHA-256,
    or downloaded with transient retries, verified, and atomically cached.
"""

from collections import OrderedDict
import hashlib
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

logger = logging.getLogger("bz_ggml.asset_cache")


class AssetCache:
    """Thread-safe bounded disk LRU cache for downloaded job input assets."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_bytes: int = 2 * 1024 * 1024 * 1024,  # 2 GiB default cap
        download_timeout_s: float = 30.0,
        max_retries: int = 3,
    ):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "bz_asset_cache")
        self.max_bytes = max(1 * 1024 * 1024, int(max_bytes))
        self.download_timeout_s = float(download_timeout_s)
        self.max_retries = max(1, int(max_retries))

        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, Tuple[str, int]]" = OrderedDict()
        self._total_bytes: int = 0
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Populates LRU index from pre-existing cache directory files."""
        with self._lock:
            items = []
            for name in os.listdir(self.cache_dir):
                if name.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(self.cache_dir, name))
                    except OSError:
                        pass
                    continue
                path = os.path.join(self.cache_dir, name)
                if os.path.isfile(path):
                    try:
                        st = os.stat(path)
                        items.append((st.st_mtime, name, path, st.st_size))
                    except OSError:
                        continue
            items.sort(key=lambda x: x[0])
            for _, key, path, size in items:
                self._entries[key] = (path, size)
                self._total_bytes += size
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self.max_bytes and self._entries:
            old_key, (old_path, old_size) = self._entries.popitem(last=False)
            self._total_bytes = max(0, self._total_bytes - old_size)
            try:
                os.remove(old_path)
            except OSError:
                pass
            logger.debug(f"[AssetCache] Evicted LRU asset {old_key} ({old_size} bytes)")

    @staticmethod
    def _normalize_sha256(sha256_hex: str) -> str:
        clean = (sha256_hex or "").strip().lower()
        if len(clean) == 64 and all(c in "0123456789abcdef" for c in clean):
            return clean
        return ""

    def resolve_bytes(
        self,
        *,
        url: str = "",
        inline_bytes: bytes = b"",
        expected_sha256: str = "",
    ) -> bytes:
        """
        Resolves an InputAsset into raw bytes.
        Priority: inline_bytes if present, else local SHA-256 LRU cache, else HTTP GET with retries.
        """
        norm_sha = self._normalize_sha256(expected_sha256)

        if inline_bytes:
            if norm_sha:
                actual = hashlib.sha256(inline_bytes).hexdigest()
                if actual != norm_sha:
                    raise ValueError(f"Inline input asset SHA-256 mismatch: expected {norm_sha}, got {actual}")
            return inline_bytes

        if not url:
            raise ValueError("InputAsset has neither inline_bytes nor url.")

        # 1. Cache lookup by SHA-256
        if norm_sha:
            with self._lock:
                cached = self._entries.get(norm_sha)
                if cached is not None:
                    path, _ = cached
                    if os.path.isfile(path):
                        self._entries.move_to_end(norm_sha)
                        with open(path, "rb") as f:
                            return f.read()
                    else:
                        _, old_size = self._entries.pop(norm_sha)
                        self._total_bytes = max(0, self._total_bytes - old_size)

        # 2. Download with transient retries
        data = self._download_with_retries(url)
        actual_sha = hashlib.sha256(data).hexdigest()
        if norm_sha and actual_sha != norm_sha:
            raise ValueError(f"Downloaded asset SHA-256 mismatch: expected {norm_sha}, got {actual_sha}")

        # 3. Store in LRU cache if size fits budget
        cache_key = norm_sha or actual_sha
        if len(data) <= self.max_bytes:
            self._put_cache(cache_key, data)

        return data

    def _put_cache(self, key: str, data: bytes) -> str:
        target_path = os.path.join(self.cache_dir, key)
        tmp_path = f"{target_path}.{threading.get_ident()}.{int(time.time() * 1000)}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, target_path)

        size = len(data)
        with self._lock:
            if key in self._entries:
                _, old_size = self._entries.pop(key)
                self._total_bytes = max(0, self._total_bytes - old_size)
            self._entries[key] = (target_path, size)
            self._total_bytes += size
            self._evict_if_needed()
        return target_path

    def _download_with_retries(self, url: str) -> bytes:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self.download_timeout_s) as resp:
                    status = getattr(resp, "status", 200)
                    if status in (200, 206):
                        return resp.read()
                    raise RuntimeError(f"Asset download returned HTTP {status}")
            except urllib.error.HTTPError as e:
                last_err = e
                # Do not retry client errors (400, 403 expired URL, 404 not found)
                if 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(f"Asset download failed with HTTP {e.code}: {e.reason}") from e
            except Exception as e:
                last_err = e

            if attempt + 1 < self.max_retries:
                backoff_s = 0.5 * (2 ** attempt)
                time.sleep(backoff_s)

        raise RuntimeError(f"Asset download failed after {self.max_retries} attempts: {last_err}")

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entry_count": len(self._entries),
                "total_bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
            }
