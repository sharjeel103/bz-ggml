"""
bz-ggml: High-Performance C++ Inference Engine for Breeze-TTS-2 on Dual GPUs with GGML
"""

from .bindings import (
    BreezeLib, GeneratorHandle, VocoderHandle,
    load_breeze_voice, save_breeze_voice,
    serialize_breeze_voice, deserialize_breeze_voice
)
from .cluster import (
    DualInstanceCluster, ClusterResult, UserTask,
    VoiceConversionTask, estimate_conversion_profile,
    DynamicCapacityCoordinator
)
from .telemetry import TelemetryProfiler
from .asr import ASRService, format_srt_time, load_audio_16k
from .audio_pipeline import (
    split_text_into_smart_chunks,
    stitch_audio_chunks,
    apply_resemble_enhancer,
    enhance_audio_array,
    apply_studio_normalization,
    master_audio_array,
    convert_to_mp3_256k,
    save_pcm_to_mp3,
    save_pcm_to_wav,
    encode_pcm_to_mp3_bytes,
    encode_pcm_to_wav_bytes,
    load_any_audio,
    generate_1word_srt_file,
    format_chunk_preview_markdown,
    PipelineJob,
    PipelineJobDispatcher,
    JobStage
)
from .orchestrator import (
    EngineOrchestrator,
    EngineCapabilities,
    TaskType,
    TaskSpec,
    Parcel,
    ParcelResult
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
    "DynamicCapacityCoordinator",
    "TelemetryProfiler",
    "ClusterTelemetry",
    "ASRService",
    "format_srt_time",
    "load_audio_16k",
    "load_breeze_voice",
    "save_breeze_voice",
    "serialize_breeze_voice",
    "deserialize_breeze_voice",
    "split_text_into_smart_chunks",
    "stitch_audio_chunks",
    "apply_resemble_enhancer",
    "enhance_audio_array",
    "apply_studio_normalization",
    "master_audio_array",
    "convert_to_mp3_256k",
    "save_pcm_to_mp3",
    "save_pcm_to_wav",
    "encode_pcm_to_mp3_bytes",
    "encode_pcm_to_wav_bytes",
    "load_any_audio",
    "generate_1word_srt_file",
    "format_chunk_preview_markdown",
    "PipelineJob",
    "PipelineJobDispatcher",
    "JobStage",
    "EngineOrchestrator",
    "EngineCapabilities",
    "TaskType",
    "TaskSpec",
    "Parcel",
    "ParcelResult"
]
