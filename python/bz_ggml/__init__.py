"""
bz-ggml: High-Performance C++ Inference Engine for Breeze-TTS-2 on Dual GPUs with GGML
"""

from .bindings import BreezeLib, GeneratorHandle, VocoderHandle
from .cluster import DualInstanceCluster, ClusterResult, UserTask
from .telemetry import TelemetryProfiler
ClusterTelemetry = TelemetryProfiler

__version__ = "1.0.0"
__all__ = [
    "BreezeLib",
    "GeneratorHandle",
    "VocoderHandle",
    "DualInstanceCluster",
    "ClusterResult",
    "UserTask",
    "TelemetryProfiler",
    "ClusterTelemetry"
]
