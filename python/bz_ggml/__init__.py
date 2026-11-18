"""
bz-ggml: High-Performance C++ Inference Engine for Breeze-TTS-2 on Dual GPUs with GGML
"""

from .bindings import (
    BreezeLib, GeneratorHandle, VocoderHandle,
    load_breeze_voice, save_breeze_voice,
    serialize_breeze_voice, deserialize_breeze_voice
)
from .cluster import DualInstanceCluster, ClusterResult, UserTask, VoiceConversionTask, estimate_conversion_profile
from .telemetry import TelemetryProfiler
from .audio_pipeline import (
    split_text_into_smart_chunks,
    stitch_audio_chunks,
    apply_resemble_enhancer,
    apply_studio_normalization,
    convert_to_mp3_256k,
    generate_1word_srt_file,
    PipelineJob,
    PipelineJobDispatcher,
    JobStage
)
ClusterTelemetry = TelemetryProfiler

__version__ = "1.0.0"
__all__ = [
    "BreezeLib",
    "GeneratorHandle",
    "VocoderHandle",
    "DualInstanceCluster",
    "ClusterResult",
    "UserTask",
    "VoiceConversionTask",
    "estimate_conversion_profile",
    "TelemetryProfiler",
    "ClusterTelemetry",
    "load_breeze_voice",
    "save_breeze_voice",
    "serialize_breeze_voice",
    "deserialize_breeze_voice",
    "split_text_into_smart_chunks",
    "stitch_audio_chunks",
    "apply_resemble_enhancer",
    "apply_studio_normalization",
    "convert_to_mp3_256k",
    "generate_1word_srt_file",
    "PipelineJob",
    "PipelineJobDispatcher",
    "JobStage"
]
