from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import secrets
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import uvicorn
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from voxcpm.dialogue_scripts import (
    delete_dialogue_script,
    load_dialogue_scripts,
    normalize_script_name,
    resolve_dialogue_script,
    save_dialogue_script,
)
from voxcpm.magic_takes import (
    MAGIC_TAKE_MEDIA_TYPE,
    decode_pcm16_mono_wav,
    pack_magic_take_bundle,
)
from voxcpm.openai_compat import (
    OPENAI_MODEL_ID,
    OPENAI_VOICE_PRESETS,
    OpenAIAPIError,
    OpenAISpeechRequest,
    normalize_openai_response_format,
    openai_error_response,
    openai_model_payload,
    openai_voice_payload,
    require_openai_api_key,
    resolve_openai_model,
    resolve_openai_voice_id,
    validate_openai_stream_format,
)
from voxcpm.standalone_ui.gpu import GPU_MONITOR
from voxcpm.standalone_ui.server import attach_ui
from voxcpm.text_planning import split_long_text
from voxcpm.ssml import (
    SSMLPlan,
    SSMLUnit,
    SSMLValidationError,
    SSMLVoiceDefinition,
    compile_ssml,
    ssml_capabilities,
)
from voxcpm.ssml_execution import (
    PreparedSSMLVoice,
    SSMLExecutionSession,
    SSMLVoiceBinding,
    assemble_audio_timeline,
)
from voxcpm.timestamps import align_audio_file, timestamp_backend_available
from voxcpm.voice_profiles import (
    delete_voice_profile,
    load_voice_profiles,
    normalize_profile_name,
    normalize_profile_recipe,
    normalize_profile_tags,
    resolve_voice_profile,
    save_voice_profile,
    update_voice_profile,
)

def _read_version() -> str:
    version_path = Path(__file__).resolve().parents[1] / "VERSION"
    if version_path.exists():
        return version_path.read_text(encoding="utf-8").strip()
    try:
        return package_version("voxcpm")
    except PackageNotFoundError:
        return "unknown"


DEFAULT_MODEL_ID = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
DEFAULT_BACKEND = os.getenv("VOXCPM_BACKEND", "native").strip().lower()
if DEFAULT_BACKEND not in {"native", "nano"}:
    raise ValueError("VOXCPM_BACKEND must be either 'native' or 'nano'")
DEFAULT_INFERENCE_TIMESTEPS = int(os.getenv("VOXCPM_NANO_INFERENCE_TIMESTEPS", "10"))
if not 1 <= DEFAULT_INFERENCE_TIMESTEPS <= 100:
    raise ValueError("VOXCPM_NANO_INFERENCE_TIMESTEPS must be between 1 and 100")
DEFAULT_DEVICE = os.getenv("VOXCPM_DEVICE", "auto")
DEFAULT_LOAD_DENOISER = os.getenv("VOXCPM_LOAD_DENOISER", "0").lower() in {"1", "true", "yes"}
DEFAULT_OPTIMIZE = os.getenv("VOXCPM_OPTIMIZE", "1").lower() not in {"0", "false", "no"}
DEFAULT_LOCAL_ONLY = os.getenv("VOXCPM_LOCAL_FILES_ONLY", "0").lower() in {"1", "true", "yes"}
ZIPENHANCER_MODEL_ID = os.getenv("ZIPENHANCER_MODEL_ID", "iic/speech_zipenhancer_ans_multiloss_16k_base")
APP_VERSION = os.getenv("APP_VERSION", _read_version())
BUILD_ID = os.getenv("BUILD_ID", "stable")
BUILD_DATE = os.getenv("VOXCPMTTS_BUILD_DATE", "unknown")
VCS_REF = os.getenv("VOXCPMTTS_VCS_REF", "unknown")
MAX_REFERENCE_UPLOAD_BYTES = int(os.getenv("VOXCPM_MAX_REFERENCE_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_MAGIC_TAKE_UPLOAD_BYTES = int(os.getenv("VOXCPM_MAX_MAGIC_TAKE_UPLOAD_BYTES", str(512 * 1024 * 1024)))
MAX_MAGIC_TAKES = 200
REFERENCE_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
REFERENCE_AUDIO_TRANSCODE_SUFFIXES = {".m4a", ".aac", ".webm"}
MAX_PORTRAIT_UPLOAD_BYTES = int(os.getenv("VOXCPM_MAX_PORTRAIT_UPLOAD_BYTES", str(5 * 1024 * 1024)))
PORTRAIT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VOICE_PROFILE_DIR = Path(os.getenv("VOXCPM_VOICE_PROFILE_DIR", "/app/persistent/voices"))
DIALOGUE_SCRIPT_DIR = Path(os.getenv("VOXCPM_DIALOGUE_SCRIPT_DIR", "/app/persistent/scripts"))
SSML_STAGING_DIR = Path(os.getenv("VOXCPM_SSML_STAGING_DIR", "/tmp/voxcpmtts-ssml"))
MAX_RANDOM_SEED = 2**32 - 1
TIMESTAMP_MODEL = os.getenv("VOXCPM_TIMESTAMP_MODEL", "base")
TIMESTAMP_DEVICE = os.getenv("VOXCPM_TIMESTAMP_DEVICE", "cpu")
LOUDNESS_NORMALIZATION_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
POST_PROCESSING_METHODS = {"ffmpeg", "signalsmith"}
POST_PROCESSING_SAMPLE_RATE = 48_000
LONG_TEXT_JOIN_SECONDS = 0.12

SUPPORTED_LANGUAGES = [
    "Arabic",
    "Burmese",
    "Chinese",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Khmer",
    "Korean",
    "Lao",
    "Malay",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Thai",
    "Turkish",
    "Vietnamese",
]

SSML_LANGUAGE_CODES = {
    "ar": "Arabic",
    "my": "Burmese",
    "zh": "Chinese",
    "yue": "Chinese",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "ms": "Malay",
    "nb": "Norwegian",
    "nn": "Norwegian",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "fil": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

OUTPUT_FORMATS = {
    "wav": {
        "label": "WAV",
        "extension": "wav",
        "media_type": "audio/wav",
        "ffmpeg_args": None,
    },
    "mp3": {
        "label": "MP3",
        "extension": "mp3",
        "media_type": "audio/mpeg",
        "ffmpeg_args": ["-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "192k"],
    },
    "opus": {
        "label": "Opus",
        "extension": "opus",
        "media_type": "audio/ogg",
        "ffmpeg_args": ["-f", "ogg", "-codec:a", "libopus", "-b:a", "128k"],
    },
    "aac": {
        "label": "AAC",
        "extension": "aac",
        "media_type": "audio/aac",
        "ffmpeg_args": ["-f", "adts", "-codec:a", "aac", "-b:a", "192k"],
    },
    "flac": {
        "label": "FLAC",
        "extension": "flac",
        "media_type": "audio/flac",
        "ffmpeg_args": ["-f", "flac", "-codec:a", "flac"],
    },
    "ogg": {
        "label": "OGG Vorbis",
        "extension": "ogg",
        "media_type": "audio/ogg",
        "ffmpeg_args": ["-f", "ogg", "-codec:a", "libvorbis", "-q:a", "5"],
    },
    "pcm": {
        "label": "PCM 24 kHz",
        "extension": "pcm",
        "media_type": "audio/pcm;rate=24000",
        "sample_rate": 24_000,
        "ui": False,
        "ffmpeg_args": ["-f", "s16le", "-codec:a", "pcm_s16le", "-ac", "1", "-ar", "24000"],
    },
}
FORMAT_ALIASES = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".ogg": "ogg",
    ".opus": "opus",
    ".aac": "aac",
    ".pcm": "pcm",
    "mpeg": "mp3",
    "vorbis": "ogg",
}
STREAM_FORMATS = {
    "mp3": {
        "label": "Progressive MP3",
        "extension": "mp3",
        "media_type": "audio/mpeg",
    },
}
STREAM_FORMAT_ALIASES = {
    ".mp3": "mp3",
    "mpeg": "mp3",
}
ASR_MODEL_ID = os.getenv("VOXCPM_ASR_MODEL_ID", "openai/whisper-base")
ASR_MODEL_REVISION = os.getenv(
    "VOXCPM_ASR_MODEL_REVISION",
    "e37978b90ca9030d5170a5c07aadb050351a65bb",
)
ASR_DEVICE = os.getenv("VOXCPM_ASR_DEVICE", "cpu")
DEFAULT_LOAD_ASR = os.getenv("VOXCPM_LOAD_ASR", "1").lower() in {"1", "true", "yes"}
ASR_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.txt",
    "LICENSE*",
    "README.md",
)

MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
MODEL_LOCK = threading.Lock()
ASR_MODEL = None
ASR_LOCK = threading.RLock()
ACTIVITY_LOCK = threading.Lock()
GENERATION_ACTIVITY: dict[str, Any] = {
    "active": False,
    "phase": "ready",
    "message": "Ready",
    "updated_at": time.time(),
}


def set_generation_activity(phase: str, message: str, *, active: bool) -> None:
    with ACTIVITY_LOCK:
        GENERATION_ACTIVITY.update(
            active=active,
            phase=phase,
            message=message,
            updated_at=time.time(),
        )


def get_generation_activity() -> dict[str, Any]:
    with ACTIVITY_LOCK:
        return dict(GENERATION_ACTIVITY)


def _close_model(model: Any) -> None:
    close = getattr(model, "close", None)
    if callable(close):
        close()


def _close_all_models() -> None:
    with MODEL_LOCK:
        for model in MODEL_CACHE.values():
            _close_model(model)
        MODEL_CACHE.clear()


def _close_asr_model() -> bool:
    global ASR_MODEL
    with ASR_LOCK:
        if ASR_MODEL is None:
            return False
        ASR_MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def get_cuda_devices() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]


def get_runtime_label() -> str:
    cuda_devices = get_cuda_devices()
    if cuda_devices:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "all")
        device_list = ", ".join(f"{idx}:{name}" for idx, name in enumerate(cuda_devices))
        return f"GPU x{len(cuda_devices)} (visible={visible}) [{device_list}]"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS"
    return "CPU"


def get_hardware_choices() -> list[tuple[str, str]]:
    choices = [("Auto", "auto"), ("CPU", "cpu")]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        choices.append(("MPS", "mps"))
    for idx, name in enumerate(get_cuda_devices()):
        choices.append((f"GPU {idx} ({name})", f"cuda:{idx}"))
    return choices


def resolve_requested_device(device: str, use_gpu: Optional[bool] = None) -> str:
    if use_gpu is True:
        return "auto"
    if use_gpu is False:
        return "cpu"
    return (device or "auto").strip().lower()


def canonical_model_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device '{device}', but CUDA is not available")
        resolved = device
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested device 'mps', but MPS is not available")
        resolved = device
    elif device == "cpu":
        resolved = device
    else:
        raise ValueError(f"Unsupported device '{device}'")
    if resolved == "cuda":
        return "cuda:0"
    return resolved


def get_model(device: str = DEFAULT_DEVICE, load_denoiser: bool = False) -> Any:
    requested_device = canonical_model_device(resolve_requested_device(device))
    should_load_denoiser = DEFAULT_LOAD_DENOISER and load_denoiser
    cache_key = (DEFAULT_BACKEND, DEFAULT_MODEL_ID, requested_device)
    with MODEL_LOCK:
        if cache_key not in MODEL_CACHE:
            if DEFAULT_BACKEND == "nano":
                from voxcpm.nano_backend import NanoVoxCPM

                model_class = NanoVoxCPM
            else:
                from voxcpm import VoxCPM

                model_class = VoxCPM
            MODEL_CACHE[cache_key] = model_class.from_pretrained(
                hf_model_id=DEFAULT_MODEL_ID,
                load_denoiser=False,
                zipenhancer_model_id=ZIPENHANCER_MODEL_ID,
                local_files_only=DEFAULT_LOCAL_ONLY,
                optimize=DEFAULT_OPTIMIZE,
                device=requested_device,
            )
        if should_load_denoiser:
            MODEL_CACHE[cache_key].enable_denoiser(ZIPENHANCER_MODEL_ID)
        return MODEL_CACHE[cache_key]


def _create_asr_pipeline(model_path: str, device: str):
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )


def get_asr_model():
    global ASR_MODEL
    if not DEFAULT_LOAD_ASR:
        raise RuntimeError("ASR is disabled. Set VOXCPM_LOAD_ASR=1 to enable reference transcription.")
    with ASR_LOCK:
        if ASR_MODEL is None:
            from huggingface_hub import snapshot_download
            device = canonical_model_device(ASR_DEVICE)
            model_path = snapshot_download(
                repo_id=ASR_MODEL_ID,
                revision=ASR_MODEL_REVISION,
                allow_patterns=list(ASR_ALLOW_PATTERNS),
                local_files_only=DEFAULT_LOCAL_ONLY,
            )
            ASR_MODEL = _create_asr_pipeline(model_path, device)
        return ASR_MODEL


def transcribe_reference_audio(audio_path: str, language: str = "auto") -> str:
    if not audio_path:
        raise ValueError("audio_path is required")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    generate_kwargs = {"task": "transcribe"}
    normalized_language = (language or "auto").strip().lower()
    if normalized_language != "auto":
        generate_kwargs["language"] = normalized_language

    with ASR_LOCK:
        result = get_asr_model()(audio_path, generate_kwargs=generate_kwargs)
    return str(result.get("text", "")).strip()


def normalize_output_format(output_format: str | None) -> str:
    normalized = (output_format or "wav").strip().lower()
    normalized = FORMAT_ALIASES.get(normalized, normalized)
    if normalized not in OUTPUT_FORMATS:
        supported = ", ".join(OUTPUT_FORMATS)
        raise ValueError(f"Unsupported output format '{output_format}'. Supported formats: {supported}")
    return normalized


def normalize_stream_format(stream_format: str | None) -> str:
    normalized = (stream_format or "mp3").strip().lower()
    normalized = STREAM_FORMAT_ALIASES.get(normalized, normalized)
    if normalized not in STREAM_FORMATS:
        supported = ", ".join(STREAM_FORMATS)
        raise ValueError(f"Unsupported stream_format '{stream_format}'. Supported formats: {supported}")
    return normalized


def to_float32_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.float32:
        return audio
    if np.issubdtype(audio.dtype, np.integer):
        return (audio.astype(np.float32) / np.iinfo(audio.dtype).max).astype(np.float32)
    return audio.astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.clip(to_float32_audio(audio), -1.0, 1.0)
    audio_int16 = (audio * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def to_int16_audio(audio: np.ndarray) -> np.ndarray:
    normalized = np.clip(to_float32_audio(np.asarray(audio)).reshape(-1), -1.0, 1.0)
    return (normalized * 32767).astype("<i2")


def normalize_audio_loudness(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    normalized = to_float32_audio(np.asarray(audio)).reshape(-1)
    if normalized.size == 0:
        return normalized
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        "-af",
        LOUDNESS_NORMALIZATION_FILTER,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=audio_to_wav_bytes(normalized, sample_rate),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for loudness normalization") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to normalize output loudness: {stderr}") from exc
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def adjust_audio_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    normalized = to_float32_audio(np.asarray(audio)).reshape(-1)
    if normalized.size == 0 or abs(speed - 1.0) < 1e-9:
        return normalized

    remaining = float(speed)
    stages: list[float] = []
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    if abs(remaining - 1.0) >= 1e-9:
        stages.append(remaining)
    audio_filter = ",".join(f"atempo={stage:.8g}" for stage in stages)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-i",
        "pipe:0",
        "-af",
        audio_filter,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=normalized.astype("<f4", copy=False).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to adjust speech speed") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to adjust speech speed: {stderr}") from exc
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def build_post_processing_filter(payload: "PostProcessingRequest") -> str:
    filters = ["highpass=f=50:p=2"]
    if payload.noise_reduction_db > 0:
        filters.append(f"afftdn=nr={payload.noise_reduction_db:g}:nf=-50:tn=1:gs=6")
    if payload.bass_db:
        filters.append(f"bass=f=110:t=q:w=0.7:g={payload.bass_db:g}:p=2:r=f32")
    if payload.presence_db:
        filters.append(f"treble=f=3500:t=q:w=0.7:g={payload.presence_db:g}:p=2:r=f32")
    if payload.dynamics:
        wet_mix = payload.dynamics / 100.0
        filters.append(
            "acompressor=threshold=0.125:ratio=2:attack=20:release=250:"
            f"knee=2.828:makeup=1.15:mix={wet_mix:g}"
        )
    filters.append("deesser=i=0.12:m=0.35:f=0.5")
    filters.append("alimiter=limit=0.841:attack=5:release=50:level=disabled")
    if payload.normalize_loudness:
        filters.append(LOUDNESS_NORMALIZATION_FILTER)
    return ",".join(filters)


def decode_post_processing_audio(source_path: str) -> np.ndarray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(POST_PROCESSING_SAMPLE_RATE),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode audio for Signalsmith processing") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Audio decoding for Signalsmith timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio for Signalsmith: {stderr}") from exc
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError("Audio decoding for Signalsmith produced no samples")
    return audio


def apply_signalsmith_voice_shape(audio: np.ndarray, payload: "PostProcessingRequest") -> np.ndarray:
    if payload.pitch_semitones == 0 and payload.speed_factor == 1:
        return np.ascontiguousarray(audio, dtype=np.float32)
    try:
        import python_stretch as signalsmith
    except ImportError as exc:
        raise RuntimeError("Signalsmith processing is unavailable because python-stretch is not installed") from exc

    try:
        processor = signalsmith.Signalsmith.Stretch(0)
        processor.preset(1, POST_PROCESSING_SAMPLE_RATE)
        processor.setTransposeSemitones(
            float(payload.pitch_semitones),
            8_000 / POST_PROCESSING_SAMPLE_RATE,
        )
        processor.timeFactor = float(payload.speed_factor)
        shaped = np.asarray(
            processor.process(np.ascontiguousarray(audio[np.newaxis, :], dtype=np.float32)),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(f"Signalsmith failed to shape the voice: {exc}") from exc
    if shaped.ndim != 2 or shaped.shape[0] != 1 or shaped.shape[1] == 0:
        raise RuntimeError("Signalsmith produced an invalid audio buffer")
    if not np.isfinite(shaped).all():
        raise RuntimeError("Signalsmith produced non-finite audio samples")
    return np.ascontiguousarray(shaped[0], dtype=np.float32)


def post_process_audio(source_path: str, payload: "PostProcessingRequest") -> bytes:
    if payload.method not in POST_PROCESSING_METHODS:
        raise ValueError(f"Unsupported post-processing method: {payload.method}")
    input_bytes: bytes | None = None
    input_args = ["-i", source_path]
    if payload.method == "signalsmith":
        decoded = decode_post_processing_audio(source_path)
        shaped = apply_signalsmith_voice_shape(decoded, payload)
        input_bytes = shaped.astype("<f4", copy=False).tobytes()
        input_args = [
            "-f",
            "f32le",
            "-ar",
            str(POST_PROCESSING_SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
        ]
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        build_post_processing_filter(payload),
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        str(POST_PROCESSING_SAMPLE_RATE),
        "-c:a",
        "pcm_f32le",
        "-f",
        "wav",
        "-threads",
        "1",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for audio post-processing") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg audio post-processing timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to post-process audio: {stderr}") from exc
    if not result.stdout:
        raise RuntimeError("ffmpeg audio post-processing produced no audio")
    return result.stdout


def encode_audio_bytes(audio: np.ndarray, output_format: str, sample_rate: int) -> bytes:
    normalized_format = normalize_output_format(output_format)
    wav_bytes = audio_to_wav_bytes(audio, sample_rate)
    ffmpeg_args = OUTPUT_FORMATS[normalized_format]["ffmpeg_args"]
    if ffmpeg_args is None:
        return wav_bytes

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        *ffmpeg_args,
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for non-WAV output formats") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode {normalized_format}: {stderr}") from exc
    return result.stdout


def encode_audio_stream(
    chunks: Iterator[np.ndarray],
    output_format: str,
    sample_rate: int,
    *,
    normalize_loudness: bool = False,
) -> Iterator[bytes]:
    normalized_format = normalize_stream_format(output_format)
    ffmpeg_args = OUTPUT_FORMATS[normalized_format]["ffmpeg_args"]
    if ffmpeg_args is None:
        raise RuntimeError(f"{normalized_format} does not support progressive streaming")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-i",
        "pipe:0",
        *(["-af", LOUDNESS_NORMALIZATION_FILTER] if normalize_loudness else []),
        *ffmpeg_args,
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg is required to stream {normalized_format}") from exc

    assert process.stdin is not None
    assert process.stdout is not None
    stop_writer = threading.Event()
    writer_errors: list[BaseException] = []

    def write_chunks() -> None:
        try:
            for chunk in chunks:
                if stop_writer.is_set():
                    break
                process.stdin.write(to_int16_audio(chunk).tobytes())
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            if not stop_writer.is_set():
                writer_errors.append(exc)
        except Exception as exc:  # Generation errors must reach the response iterator.
            writer_errors.append(exc)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            close = getattr(chunks, "close", None)
            if callable(close):
                close()

    writer = threading.Thread(target=write_chunks, name="voxcpmtts-stream-encoder", daemon=True)
    writer.start()
    try:
        while True:
            data = process.stdout.read(65536)
            if data:
                yield data
                continue
            if process.poll() is not None:
                break
    finally:
        stop_writer.set()
        try:
            process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
        writer.join(timeout=5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)

    stderr = process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr else ""
    if writer_errors:
        raise RuntimeError(f"Audio streaming failed: {writer_errors[0]}")
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to stream {normalized_format}: {stderr}")


def clean_control(control: str | None) -> str:
    control = (control or "").strip()
    return control.replace("(", "").replace(")", "").replace("（", "").replace("）", "").strip()


def build_final_text(text: str, control: str | None, prompt_text: str | None) -> str:
    stripped = (text or "").strip()
    control = clean_control(control)
    if control and not prompt_text:
        return f"({control}){stripped}"
    return stripped


def resolve_generation_seed(payload: "TTSRequest") -> int:
    if payload.randomize_seed:
        return secrets.randbelow(MAX_RANDOM_SEED + 1)
    return 42 if payload.seed is None else payload.seed


def build_generate_kwargs(payload: "TTSRequest", model: Any, *, seed: int | None = None) -> dict:
    ref_audio = payload.ref_audio or payload.reference_audio
    prompt_audio = payload.prompt_audio
    prompt_text = payload.prompt_text
    control = payload.control or payload.instruct
    if payload.voice_profile:
        _, profile = resolve_voice_profile(VOICE_PROFILE_DIR, payload.voice_profile)
        if not ref_audio and not prompt_audio:
            ref_audio = profile.get("ref_audio") or None
        if payload.clone_mode != "reference" and not prompt_text and not payload.ref_text:
            prompt_text = profile.get("ref_text") or None
        if not control:
            control = profile.get("control") or None
    if payload.ref_text and not prompt_text:
        prompt_text = payload.ref_text
    if ref_audio and prompt_text and not prompt_audio:
        prompt_audio = ref_audio
    if payload.clone_mode == "reference":
        prompt_audio = None
        prompt_text = None
    elif payload.clone_mode == "transcript" and not prompt_text:
        raise ValueError("Transcript-guided cloning requires a reference transcript.")

    final_text = build_final_text(payload.text, control, prompt_text)
    kwargs = {
        "text": final_text,
        "reference_wav_path": ref_audio,
        "prompt_wav_path": prompt_audio,
        "prompt_text": prompt_text,
        "cfg_value": payload.cfg_value,
        "inference_timesteps": payload.inference_timesteps,
        "normalize": payload.normalize_text,
        "denoise": payload.denoise,
        "seed": resolve_generation_seed(payload) if seed is None else seed,
    }
    if not prompt_audio:
        kwargs["prompt_text"] = None
    if not ref_audio:
        kwargs["reference_wav_path"] = None
    return kwargs


class TTSRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1, description="Text to synthesize.")
    input_type: Literal["text", "ssml", "ssml-h"] = Field(
        "text",
        description="Explicit input format. Markup is parsed only for ssml or ssml-h.",
    )
    language: str = Field("English", description="Compatibility hint; VoxCPM2 auto-detects supported languages.")
    voice: str = Field("auto", description="Compatibility field. Use instruct/control or reference audio for VoxCPM voices.")
    voice_profile: Optional[str] = Field(None, description="Saved local voice profile name.")
    clone_mode: Literal["auto", "reference", "transcript"] = Field(
        "auto",
        description="Clone conditioning: automatic, reference-only direction, or transcript-guided.",
    )
    control: Optional[str] = Field(None, description="VoxCPM voice design/control instruction.")
    instruct: Optional[str] = Field(None, description="Alias for control.")
    reference_audio: Optional[str] = Field(None, description="Container-visible reference audio path.")
    ref_audio: Optional[str] = Field(None, description="Alias for reference_audio.")
    ref_text: Optional[str] = Field(None, description="Transcript of the reference audio for ultimate cloning.")
    prompt_audio: Optional[str] = Field(None, description="Prompt audio path for continuation cloning.")
    prompt_text: Optional[str] = Field(None, description="Transcript for prompt_audio.")
    cfg_value: float = Field(2.0, ge=0.1, le=10.0, description="Classifier-free guidance scale.")
    inference_timesteps: int = Field(
        DEFAULT_INFERENCE_TIMESTEPS,
        ge=1,
        le=100,
        description="LocDiT flow-matching steps. Nano-vLLM fixes this value when the engine starts.",
    )
    normalize_text: bool = Field(False, alias="normalize", description="Normalize text before synthesis.")
    normalize_loudness: bool = Field(
        False,
        description="Normalize output toward -16 LUFS with a -1.5 dB true-peak ceiling.",
    )
    protect_long_audio: bool = Field(
        True,
        description="Split long text near sentence boundaries to limit accumulated ringing and degradation.",
    )
    denoise: bool = Field(False, description="Apply ZipEnhancer to prompt/reference audio when denoiser is enabled.")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Compatibility field; VoxCPM2 has no direct speed scalar.")
    seed: Optional[int] = Field(42, ge=0, le=MAX_RANDOM_SEED, description="32-bit generation seed.")
    randomize_seed: bool = Field(True, description="Choose a fresh generation seed for this request.")
    device: str = Field("auto", description="auto, cpu, mps, cuda, or cuda:N.")
    use_gpu: Optional[bool] = Field(None, description="Legacy compatibility switch. Prefer device.")
    output_format: str = Field(
        "wav",
        alias="format",
        description="Response audio format. Supported: wav, mp3, flac, ogg.",
    )


class StreamingTTSRequest(TTSRequest):
    stream_format: str = Field("mp3", description="Progressive response format. Supported: mp3.")


class MagicTakeRequest(BaseModel):
    payload: TTSRequest
    speech_index: int = Field(..., ge=0, lt=MAX_MAGIC_TAKES)


class MagicTimelineItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["speech", "break"] = Field(..., alias="type")
    id: str | None = None
    milliseconds: int | None = Field(None, ge=0, le=10_000)


class MagicAssemblyRequest(BaseModel):
    items: list[MagicTimelineItem] = Field(..., min_length=1, max_length=300)
    output_format: str = "wav"
    normalize_loudness: bool = False


def planned_text_sections(payload: TTSRequest) -> list[str]:
    text = payload.text.strip()
    if not payload.protect_long_audio:
        return [text] if text else []
    return split_long_text(text)


def generate_planned_waveform(payload: TTSRequest, model: Any, seed: int) -> np.ndarray:
    sections = planned_text_sections(payload)
    generated: list[np.ndarray] = []
    sample_rate = int(model.tts_model.sample_rate)
    silence = np.zeros(round(sample_rate * LONG_TEXT_JOIN_SECONDS), dtype=np.float32)
    for index, section in enumerate(sections):
        if len(sections) > 1:
            set_generation_activity(
                "generating",
                f"Generating protected section {index + 1} of {len(sections)}",
                active=True,
            )
        section_payload = payload.model_copy(update={"text": section})
        generated.append(to_float32_audio(model.generate(**build_generate_kwargs(section_payload, model, seed=seed))))
        if index + 1 < len(sections):
            generated.append(silence)
    return np.concatenate(generated) if generated else np.zeros(0, dtype=np.float32)


def iter_planned_waveform_chunks(
    payload: StreamingTTSRequest,
    model: Any,
    seed: int,
) -> Iterator[np.ndarray]:
    sections = planned_text_sections(payload)
    sample_rate = int(model.tts_model.sample_rate)
    for index, section in enumerate(sections):
        if len(sections) > 1:
            set_generation_activity(
                "streaming",
                f"Streaming protected section {index + 1} of {len(sections)}",
                active=True,
            )
        section_payload = payload.model_copy(update={"text": section})
        chunks = model.generate_streaming(**build_generate_kwargs(section_payload, model, seed=seed))
        try:
            yield from chunks
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
        if index + 1 < len(sections):
            yield np.zeros(round(sample_rate * LONG_TEXT_JOIN_SECONDS), dtype=np.float32)


def _openai_profile(profile_name: str, *, required: bool) -> tuple[str | None, dict[str, Any] | None]:
    try:
        normalized_name = normalize_profile_name(profile_name)
        return resolve_voice_profile(VOICE_PROFILE_DIR, normalized_name)
    except ValueError as exc:
        if required:
            raise OpenAIAPIError(
                f"Voice '{profile_name}' is not available.",
                param="voice",
                code="unsupported_voice",
            ) from exc
        return None, None


def openai_speech_to_tts_request(payload: OpenAISpeechRequest) -> TTSRequest:
    resolve_openai_model(payload.model)
    validate_openai_stream_format(payload.stream_format)
    output_format = normalize_openai_response_format(payload.response_format)
    if payload.clone_mode not in {"auto", "reference", "transcript"}:
        raise OpenAIAPIError(
            "clone_mode must be auto, reference, or transcript.",
            param="clone_mode",
            code="invalid_parameter",
        )

    voice_id = resolve_openai_voice_id(payload.voice)
    if voice_id.lower() == "default":
        voice_id = "alloy"
    explicit_profile = (payload.voice_profile or "").strip()
    profile_name: str | None = None
    profile: dict[str, Any] | None = None
    preset = OPENAI_VOICE_PRESETS.get(voice_id.lower())
    if explicit_profile:
        profile_name, profile = _openai_profile(explicit_profile, required=True)
    elif not preset and voice_id.lower() not in {"auto", "default"}:
        profile_name, profile = _openai_profile(voice_id, required=True)

    recipe = dict((profile or {}).get("recipe") or {})
    fields_set = payload.model_fields_set
    instructions = (payload.instructions or "").strip()
    control = instructions or (preset[0] if preset else "")
    language = payload.language
    if "language" not in fields_set and profile and profile.get("language"):
        language = str(profile["language"])

    if payload.randomize_seed is not None:
        randomize_seed = payload.randomize_seed
    elif profile and isinstance(recipe.get("randomize_seed"), bool):
        randomize_seed = bool(recipe["randomize_seed"])
    else:
        randomize_seed = voice_id.lower() in {"auto", "default"} and not preset

    seed = payload.seed
    if seed is None and not randomize_seed:
        if profile and isinstance(recipe.get("seed"), int):
            seed = int(recipe["seed"])
        elif preset:
            seed = preset[1]
        else:
            seed = 42

    def recipe_value(field: str, fallback: Any) -> Any:
        return getattr(payload, field) if field in fields_set else recipe.get(field, fallback)

    clone_mode = payload.clone_mode
    if "clone_mode" not in fields_set and recipe.get("clone_mode") in {"reference", "transcript"}:
        clone_mode = str(recipe["clone_mode"])

    return TTSRequest(
        text=payload.input,
        input_type="text",
        language=language,
        voice=voice_id,
        voice_profile=None if payload.ref_audio else profile_name,
        clone_mode=clone_mode,
        control=control or None,
        ref_audio=payload.ref_audio,
        ref_text=payload.ref_text,
        cfg_value=float(recipe_value("cfg_value", payload.cfg_value)),
        inference_timesteps=int(
            recipe_value("inference_timesteps", payload.inference_timesteps or DEFAULT_INFERENCE_TIMESTEPS)
        ),
        normalize_text=bool(
            payload.normalize_text
            if "normalize_text" in fields_set
            else recipe.get("normalize", payload.normalize_text)
        ),
        normalize_loudness=bool(recipe_value("normalize_loudness", payload.normalize_loudness)),
        protect_long_audio=bool(recipe_value("protect_long_audio", payload.protect_long_audio)),
        denoise=bool(recipe_value("denoise", payload.denoise)),
        speed=payload.speed,
        seed=seed,
        randomize_seed=randomize_seed,
        device=payload.device,
        output_format=output_format,
    )


class PostProcessingRequest(BaseModel):
    method: Literal["ffmpeg", "signalsmith"] = Field("ffmpeg", description="Audio finishing backend.")
    preset: Literal["clean", "studio", "custom"] = Field("studio", description="UI preset identity.")
    pitch_semitones: float = Field(0.0, ge=-6.0, le=6.0, description="Signalsmith pitch shift in semitones.")
    speed_factor: float = Field(1.0, ge=0.75, le=1.25, description="Signalsmith playback-speed factor.")
    noise_reduction_db: float = Field(2.0, ge=0.0, le=12.0, description="Steady-noise attenuation in dB.")
    bass_db: float = Field(1.0, ge=-6.0, le=6.0, description="Low-shelf gain around 110 Hz.")
    presence_db: float = Field(1.0, ge=-6.0, le=6.0, description="High-shelf gain around 3.5 kHz.")
    dynamics: int = Field(35, ge=0, le=100, description="Parallel compression amount from 0 to 100.")
    normalize_loudness: bool = Field(True, description="Finish toward -16 LUFS and -1.5 dBTP.")


class MetricsRequest(BaseModel):
    text: str = Field("", description="Text to inspect.")
    input_type: Literal["text", "ssml", "ssml-h"] = "text"
    language: str = "English"


class DialogueScriptRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    document: str = Field(..., min_length=1, max_length=1_000_000)
    description: str = Field("", max_length=240)
    tags: list[str] = Field(default_factory=list, max_length=12)


class PurgeRequest(BaseModel):
    device: Optional[str] = Field(None, description="Optional cached device to clear. Omit to clear all cached models.")


class TranscriptionRequest(BaseModel):
    audio_path: str = Field(..., description="Container-visible audio path to transcribe.")
    language: str = Field("auto", description="ASR language hint. Use auto for detection.")


def resolve_ssml_language(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("SSML language must not be empty.")
    for language in SUPPORTED_LANGUAGES:
        if normalized.casefold() == language.casefold():
            return language
    code = normalized.replace("_", "-").split("-", 1)[0].lower()
    language = SSML_LANGUAGE_CODES.get(code)
    if language is None:
        raise ValueError(f"Unsupported SSML language '{value}'.")
    return language


def build_ssml_h_voice_control(definition: SSMLVoiceDefinition) -> str:
    values: list[str] = []
    if definition.description:
        values.append(definition.description.strip())
    for label, value in (
        ("gender", definition.gender),
        ("age", definition.age),
        ("pitch", definition.pitch),
        ("style", definition.style),
        ("accent", definition.accent),
        ("dialect", definition.dialect),
    ):
        if value:
            values.append(f"{value.replace('-', ' ')} {label}")
    if definition.languages:
        languages = [resolve_ssml_language(value) for value in definition.languages]
        values.append(f"speaking {', '.join(languages)}")
    if not values:
        raise SSMLValidationError(
            f"SSML-H voice '{definition.name}' needs h:description or at least one voice-design attribute."
        )
    return ", ".join(dict.fromkeys(values))


def compile_ssml_request(payload: TTSRequest) -> tuple[SSMLPlan, dict[str, dict[str, Any]]]:
    if payload.normalize_text:
        raise SSMLValidationError("normalize_text is available only when input_type='text'.")
    profiles = load_voice_profiles(VOICE_PROFILE_DIR)
    default_language = None
    if (payload.language or "").strip().lower() != "auto":
        default_language = resolve_ssml_language(payload.language)

    def validate_voice(name: str, _definitions: frozenset[str]) -> None:
        resolve_voice_profile(VOICE_PROFILE_DIR, name)

    plan = compile_ssml(
        payload.text,
        payload.input_type,
        default_language=default_language,
        resolve_language=resolve_ssml_language,
        validate_voice=validate_voice,
    )
    for index, unit in enumerate(plan.units):
        if unit.kind != "speech":
            continue
        effective_rate = payload.speed * unit.prosody.rate
        if not 0.25 <= effective_rate <= 4.0:
            raise SSMLValidationError(
                f"Effective rate for SSML unit {index + 1} must be between 0.25 and 4.0."
            )
        if not -12.0 <= unit.prosody.pitch_semitones <= 12.0:
            raise SSMLValidationError(
                f"Effective pitch for SSML unit {index + 1} must be between -12st and +12st."
            )
        if not 0.0 <= unit.prosody.volume <= 2.0:
            raise SSMLValidationError(
                f"Effective volume for SSML unit {index + 1} must be between 0 and 2.0."
            )
    for definition in plan.voice_definitions:
        profile_name = normalize_profile_name(definition.name)
        if profile_name in profiles and not (definition.scope == "profile" and definition.replace):
            raise SSMLValidationError(
                f"Voice profile '{profile_name}' already exists. Use scope='profile' replace='true' to replace it."
            )
        if definition.sample_language:
            resolve_ssml_language(definition.sample_language)
        build_ssml_h_voice_control(definition)
    return plan, profiles


def _binding_from_profile(name: str, profiles: dict[str, dict[str, Any]]) -> SSMLVoiceBinding:
    profile_name = normalize_profile_name(name)
    if profile_name not in profiles:
        raise ValueError(f"Voice profile '{profile_name}' does not exist.")
    _, profile = resolve_voice_profile(VOICE_PROFILE_DIR, profile_name)
    return SSMLVoiceBinding(
        name=profile_name,
        ref_audio=profile.get("ref_audio") or None,
        ref_text=profile.get("ref_text") or None,
        control=profile.get("control") or None,
        language=profile.get("language") or None,
    )


def _default_ssml_binding(payload: TTSRequest, profiles: dict[str, dict[str, Any]]) -> SSMLVoiceBinding:
    if payload.voice_profile:
        return _binding_from_profile(payload.voice_profile, profiles)
    ref_audio = payload.ref_audio or payload.reference_audio or payload.prompt_audio
    ref_text = payload.ref_text or payload.prompt_text
    return SSMLVoiceBinding(
        ref_audio=ref_audio,
        ref_text=ref_text,
        control=payload.control or payload.instruct,
        language=payload.language,
    )


def create_ssml_execution_session(payload: TTSRequest, used_seed: int) -> SSMLExecutionSession:
    plan, profiles = compile_ssml_request(payload)
    requested_device = resolve_requested_device(payload.device, payload.use_gpu)
    canonical_device = canonical_model_device(requested_device)
    cache_key = (DEFAULT_BACKEND, DEFAULT_MODEL_ID, canonical_device)
    with MODEL_LOCK:
        model_loaded = cache_key in MODEL_CACHE
    set_generation_activity(
        "preparing" if model_loaded else "loading_model",
        "Preparing the loaded model" if model_loaded else "Loading VoxCPM2 weights",
        active=True,
    )
    model = get_model(requested_device, load_denoiser=payload.denoise)
    sample_rate = int(model.tts_model.sample_rate)

    def resolve_voice(name: str) -> SSMLVoiceBinding:
        return _binding_from_profile(name, profiles)

    def generate(
        text: str,
        binding: SSMLVoiceBinding,
        language: str | None,
        seed: int,
    ) -> np.ndarray:
        unit_payload = TTSRequest(
            text=text,
            input_type="text",
            language=language or binding.language or payload.language,
            control=binding.control,
            ref_audio=binding.ref_audio,
            ref_text=binding.ref_text,
            cfg_value=payload.cfg_value,
            inference_timesteps=payload.inference_timesteps,
            normalize_text=False,
            protect_long_audio=payload.protect_long_audio,
            denoise=payload.denoise,
            seed=seed,
            randomize_seed=False,
            device=payload.device,
            use_gpu=payload.use_gpu,
            output_format="wav",
        )
        return generate_planned_waveform(unit_payload, model, seed)

    def prepare_voice(
        definition: SSMLVoiceDefinition,
        sample_text: str,
        sample_language: str | None,
        seed: int,
        staging_dir: Path,
    ) -> SSMLVoiceBinding:
        set_generation_activity("generating", f"Designing SSML-H voice {definition.name}", active=True)
        control = build_ssml_h_voice_control(definition)
        waveform = generate(
            sample_text,
            SSMLVoiceBinding(control=control, language=sample_language),
            sample_language,
            seed,
        )
        audio_path = staging_dir / f"{normalize_profile_name(definition.name)}.wav"
        audio_path.write_bytes(audio_to_wav_bytes(waveform, sample_rate))
        return SSMLVoiceBinding(
            name=definition.name,
            ref_audio=str(audio_path),
            ref_text=sample_text,
            language=sample_language,
            generation_seed=seed,
        )

    def generate_speech(
        unit: SSMLUnit,
        binding: SSMLVoiceBinding,
        seed: int,
    ) -> tuple[int, np.ndarray]:
        set_generation_activity("generating", "Generating SSML speech", active=True)
        language = unit.language
        if not unit.language_explicit and binding.language:
            language = binding.language
        return sample_rate, generate(unit.text, binding, language, seed)

    def commit_profiles(prepared: list[PreparedSSMLVoice], _staging_dir: Path) -> dict[str, str]:
        committed: dict[str, str] = {}
        for item in prepared:
            profile_name, _ = save_voice_profile(
                VOICE_PROFILE_DIR,
                name=item.definition.name,
                profile_type="cloned",
                source_audio=item.binding.ref_audio,
                ref_text=item.sample_text,
                control=build_ssml_h_voice_control(item.definition),
                description=(item.definition.description or f"SSML-H designed voice {item.definition.name}")[:240],
                language=item.sample_language or item.binding.language or "",
                overwrite=item.definition.replace,
            )
            committed[item.definition.name] = profile_name
        return committed

    session = SSMLExecutionSession(
        plan=plan,
        default_binding=_default_ssml_binding(payload, profiles),
        request_seed=used_seed,
        request_speed=payload.speed,
        default_sample_rate=sample_rate,
        staging_parent=SSML_STAGING_DIR,
        resolve_voice=resolve_voice,
        resolve_language=resolve_ssml_language,
        prepare_voice=prepare_voice,
        generate_speech=generate_speech,
        commit_profiles=commit_profiles,
    )
    try:
        session.prepare()
    except Exception:
        session.close()
        raise
    return session


def _validated_magic_speech_ids(plan: SSMLPlan, speech_ids: list[str]) -> list[SSMLUnit]:
    speech_units = [unit for unit in plan.units if unit.kind == "speech"]
    if not speech_units:
        raise ValueError("Magic documents must contain at least one speech turn.")
    if len(speech_units) != len(speech_ids):
        raise ValueError("Magic speech IDs do not match the compiled speech turns.")
    if len(speech_ids) > MAX_MAGIC_TAKES:
        raise ValueError(f"Magic supports at most {MAX_MAGIC_TAKES} speech takes per render.")
    if len(set(speech_ids)) != len(speech_ids):
        raise ValueError("Magic speech IDs must be unique.")
    if any(not value or len(value) > 64 or not value.replace("-", "").replace("_", "").isalnum() for value in speech_ids):
        raise ValueError("Magic speech IDs may contain only letters, numbers, hyphens, and underscores.")
    return speech_units


def _magic_timeline(plan: SSMLPlan, takes: dict[str, np.ndarray], speech_ids: list[str]) -> Iterator[tuple[str, Any]]:
    speech_index = 0
    for unit in plan.units:
        if unit.kind == "break":
            yield "break", unit.duration_ms
            continue
        take_id = speech_ids[speech_index]
        yield "speech", takes[take_id]
        speech_index += 1


def render_magic_take_bundle(
    payload: TTSRequest,
    speech_ids: list[str],
    retained: dict[str, tuple[int, int, np.ndarray]],
) -> tuple[bytes, int, str, int]:
    if payload.input_type != "ssml-h":
        raise ValueError("Magic take rendering requires input_type='ssml-h'.")
    output_format = normalize_output_format(payload.output_format)
    request_seed = resolve_generation_seed(payload)
    plan, _ = compile_ssml_request(payload)
    _validated_magic_speech_ids(plan, speech_ids)
    unknown = set(retained) - set(speech_ids)
    if unknown:
        raise ValueError(f"Retained Magic take does not belong to this document: {sorted(unknown)[0]}")

    takes: dict[str, np.ndarray] = {}
    take_seeds: dict[str, int] = {}
    sample_rates = {sample_rate for _, sample_rate, _ in retained.values()}
    if len(sample_rates) > 1:
        raise ValueError("Retained Magic takes use inconsistent sample rates.")
    session: SSMLExecutionSession | None = None
    try:
        if len(retained) != len(speech_ids):
            session = create_ssml_execution_session(payload, request_seed)
            plan = session.plan
            _validated_magic_speech_ids(plan, speech_ids)
            expected_rate = session.default_sample_rate
            if sample_rates and sample_rates != {expected_rate}:
                raise ValueError("Retained Magic takes do not match the model sample rate.")
            for speech_index, take_id in enumerate(speech_ids):
                if take_id in retained:
                    seed, _, waveform = retained[take_id]
                else:
                    _, waveform, seed = session.render_speech_take(speech_index)
                takes[take_id] = waveform
                take_seeds[take_id] = seed
            sample_rate = expected_rate
            session.commit_profiles()
        else:
            sample_rate = next(iter(sample_rates), 0)
            if not sample_rate:
                raise ValueError("Magic render has no audio takes to assemble.")
            for take_id, (seed, _, waveform) in retained.items():
                takes[take_id] = waveform
                take_seeds[take_id] = seed

        waveform = assemble_audio_timeline(_magic_timeline(plan, takes, speech_ids), sample_rate)
        if payload.normalize_loudness:
            waveform = normalize_audio_loudness(waveform, sample_rate)
        output = encode_audio_bytes(waveform, output_format, sample_rate)
        packed_takes = [
            (take_id, take_seeds[take_id], audio_to_wav_bytes(takes[take_id], sample_rate))
            for take_id in speech_ids
        ]
        bundle = pack_magic_take_bundle(
            packed_takes,
            output,
            sample_rate=sample_rate,
            output_format=output_format,
            output_media_type=OUTPUT_FORMATS[output_format]["media_type"],
            request_seed=request_seed,
        )
        return bundle, request_seed, output_format, sample_rate
    finally:
        if session is not None:
            session.close()


def render_magic_take(payload: TTSRequest, speech_index: int) -> tuple[bytes, int, int]:
    if payload.input_type != "ssml-h":
        raise ValueError("Magic take rendering requires input_type='ssml-h'.")
    request_seed = resolve_generation_seed(payload)
    session = create_ssml_execution_session(payload, request_seed)
    try:
        sample_rate, waveform, take_seed = session.render_speech_take(speech_index)
        return audio_to_wav_bytes(waveform, sample_rate), take_seed, sample_rate
    finally:
        session.close()


def assemble_magic_uploads(
    payload: MagicAssemblyRequest,
    takes: dict[str, tuple[int, np.ndarray]],
) -> tuple[bytes, str, int]:
    output_format = normalize_output_format(payload.output_format)
    sample_rates = {sample_rate for sample_rate, _ in takes.values()}
    if len(sample_rates) != 1:
        raise ValueError("Magic takes must use one consistent sample rate.")
    sample_rate = next(iter(sample_rates))
    used: set[str] = set()
    total_break_ms = 0
    timeline: list[tuple[str, Any]] = []
    for item in payload.items:
        if item.kind == "speech":
            if not item.id or item.id not in takes:
                raise ValueError(f"Magic timeline references a missing speech take: {item.id or '(empty)'}.")
            used.add(item.id)
            timeline.append(("speech", takes[item.id][1]))
        else:
            duration = item.milliseconds if item.milliseconds is not None else 0
            total_break_ms += duration
            if total_break_ms > 60_000:
                raise ValueError("Magic timeline exceeds the 60000ms total break limit.")
            timeline.append(("break", duration))
    if not used:
        raise ValueError("Magic timeline must contain at least one speech take.")
    waveform = assemble_audio_timeline(iter(timeline), sample_rate)
    if payload.normalize_loudness:
        waveform = normalize_audio_loudness(waveform, sample_rate)
    return encode_audio_bytes(waveform, output_format, sample_rate), output_format, sample_rate


def synthesize_payload(payload: TTSRequest) -> tuple[str, int, np.ndarray, int]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    try:
        output_format = normalize_output_format(payload.output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    requested_device = resolve_requested_device(payload.device, payload.use_gpu)
    try:
        canonical_device = canonical_model_device(requested_device)
        cache_key = (DEFAULT_BACKEND, DEFAULT_MODEL_ID, canonical_device)
        with MODEL_LOCK:
            model_loaded = cache_key in MODEL_CACHE
        if model_loaded:
            set_generation_activity("preparing", "Preparing the loaded model", active=True)
        else:
            set_generation_activity("loading_model", "Loading VoxCPM2 weights", active=True)
        model = get_model(requested_device, load_denoiser=payload.denoise)
        set_generation_activity("generating", "Generating speech", active=True)
        seed = resolve_generation_seed(payload)
        wav = generate_planned_waveform(payload, model, seed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"VoxCPM generation failed: {exc}") from exc

    successful_seed = getattr(model, "last_successful_seed", None)
    if successful_seed is None:
        successful_seed = getattr(model.tts_model, "last_successful_seed", None)
    if successful_seed is None:
        successful_seed = seed
    sample_rate = int(model.tts_model.sample_rate)
    waveform = to_float32_audio(wav)
    if abs(payload.speed - 1.0) >= 1e-9:
        try:
            waveform = adjust_audio_speed(waveform, sample_rate, payload.speed)
        except RuntimeError as exc:
            set_generation_activity("failed", str(exc), active=False)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return output_format, sample_rate, waveform, int(successful_seed)


def synthesize_payload_chunks(payload: StreamingTTSRequest) -> tuple[str, int, Iterator[np.ndarray], int]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    try:
        output_format = normalize_stream_format(payload.stream_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    requested_device = resolve_requested_device(payload.device, payload.use_gpu)
    try:
        canonical_device = canonical_model_device(requested_device)
        cache_key = (DEFAULT_BACKEND, DEFAULT_MODEL_ID, canonical_device)
        with MODEL_LOCK:
            model_loaded = cache_key in MODEL_CACHE
        if model_loaded:
            set_generation_activity("preparing", "Preparing the loaded model", active=True)
        else:
            set_generation_activity("loading_model", "Loading VoxCPM2 weights", active=True)
        model = get_model(requested_device, load_denoiser=payload.denoise)
        set_generation_activity("streaming", "Generating the live audio stream", active=True)
        seed = resolve_generation_seed(payload)
        chunks = iter_planned_waveform_chunks(payload, model, seed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"VoxCPM streaming failed: {exc}") from exc

    return output_format, int(model.tts_model.sample_rate), chunks, seed


def _ssml_headers(
    payload: TTSRequest,
    route_name: str,
    output_format: str,
    sample_rate: int,
    seed: int,
    *,
    streaming: bool,
    duration: float | None = None,
) -> dict[str, str]:
    extension = (STREAM_FORMATS if streaming else OUTPUT_FORMATS)[output_format]["extension"]
    headers = {
        "Content-Disposition": (
            f"inline; filename=voxcpm-ssml.{extension}"
            if streaming
            else f"attachment; filename=voxcpm-ssml.{extension}"
        ),
        "X-VoxCPM-Model": DEFAULT_MODEL_ID,
        "X-VoxCPM-Sample-Rate": str(
            (STREAM_FORMATS if streaming else OUTPUT_FORMATS)[output_format].get("sample_rate", sample_rate)
        ),
        "X-VoxCPM-Route": route_name,
        "X-VoxCPM-Format": output_format,
        "X-VoxCPM-Seed": str(seed),
        "X-VoxCPM-Input-Type": payload.input_type,
        "X-VoxCPM-Loudness-Normalized": str(payload.normalize_loudness).lower(),
        "X-VoxCPM-Long-Text-Protected": str(payload.protect_long_audio).lower(),
    }
    if streaming:
        headers["X-VoxCPM-Streaming"] = "ssml-units"
    if duration is not None:
        headers["X-VoxCPM-Duration"] = f"{duration:.3f}"
    return headers


def ssml_audio_response(payload: TTSRequest, route_name: str) -> StreamingResponse:
    session: SSMLExecutionSession | None = None
    try:
        output_format = normalize_output_format(payload.output_format)
        seed = resolve_generation_seed(payload)
        session = create_ssml_execution_session(payload, seed)
        sample_rate, waveform = session.render_array()
        set_generation_activity("encoding", f"Encoding {output_format.upper()} audio", active=True)
        if payload.normalize_loudness:
            waveform = normalize_audio_loudness(waveform, sample_rate)
        audio_bytes = encode_audio_bytes(waveform, output_format, sample_rate)
        session.commit_profiles()
        set_generation_activity("complete", "Audio is ready", active=False)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=OUTPUT_FORMATS[output_format]["media_type"],
            headers=_ssml_headers(
                payload,
                route_name,
                output_format,
                sample_rate,
                seed,
                streaming=False,
                duration=len(waveform) / sample_rate if sample_rate else 0,
            ),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"SSML generation failed: {exc}") from exc
    finally:
        if session is not None:
            session.close()


def ssml_progressive_audio_response(
    payload: StreamingTTSRequest,
    route_name: str,
    *,
    cleanup_paths: tuple[str, ...] = (),
) -> StreamingResponse:
    try:
        output_format = normalize_stream_format(payload.stream_format)
        seed = resolve_generation_seed(payload)
        session = create_ssml_execution_session(payload, seed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        for path in cleanup_paths:
            Path(path).unlink(missing_ok=True)
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        for path in cleanup_paths:
            Path(path).unlink(missing_ok=True)
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"SSML streaming failed: {exc}") from exc

    sample_rate = session.default_sample_rate

    def body() -> Iterator[bytes]:
        completed = False
        try:
            yield from encode_audio_stream(
                session.iter_chunks(),
                output_format,
                sample_rate,
                normalize_loudness=payload.normalize_loudness,
            )
            session.commit_profiles()
            completed = True
            set_generation_activity("complete", "Stream is ready", active=False)
        except Exception as exc:
            set_generation_activity("failed", str(exc), active=False)
            raise
        finally:
            if not completed:
                set_generation_activity("failed", "SSML stream ended before completion", active=False)
            session.close()
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)

    return StreamingResponse(
        body(),
        media_type=STREAM_FORMATS[output_format]["media_type"],
        headers=_ssml_headers(
            payload,
            route_name,
            output_format,
            sample_rate,
            seed,
            streaming=True,
        ),
    )


def stream_audio_response(payload: TTSRequest, route_name: str) -> StreamingResponse:
    if payload.input_type != "text":
        return ssml_audio_response(payload, route_name)
    output_format, sample_rate, waveform, seed = synthesize_payload(payload)
    try:
        set_generation_activity("encoding", f"Encoding {output_format.upper()} audio", active=True)
        if payload.normalize_loudness:
            waveform = normalize_audio_loudness(waveform, sample_rate)
        audio_bytes = encode_audio_bytes(waveform, output_format, sample_rate)
    except RuntimeError as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    set_generation_activity("complete", "Audio is ready", active=False)

    extension = OUTPUT_FORMATS[output_format]["extension"]
    media_type = OUTPUT_FORMATS[output_format]["media_type"]
    duration = len(waveform) / sample_rate if sample_rate else 0
    headers = {
        "Content-Disposition": f"attachment; filename=voxcpm.{extension}",
        "X-VoxCPM-Model": DEFAULT_MODEL_ID,
        "X-VoxCPM-Sample-Rate": str(OUTPUT_FORMATS[output_format].get("sample_rate", sample_rate)),
        "X-VoxCPM-Duration": f"{duration:.3f}",
        "X-VoxCPM-Route": route_name,
        "X-VoxCPM-Format": output_format,
        "X-VoxCPM-Seed": str(seed),
        "X-VoxCPM-Input-Type": payload.input_type,
        "X-VoxCPM-Loudness-Normalized": str(payload.normalize_loudness).lower(),
        "X-VoxCPM-Long-Text-Protected": str(payload.protect_long_audio).lower(),
    }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type, headers=headers)


def progressive_audio_response(
    payload: StreamingTTSRequest,
    route_name: str,
    *,
    cleanup_paths: tuple[str, ...] = (),
) -> StreamingResponse:
    if payload.input_type != "text":
        return ssml_progressive_audio_response(payload, route_name, cleanup_paths=cleanup_paths)
    try:
        output_format, sample_rate, chunks, seed = synthesize_payload_chunks(payload)
    except Exception:
        for path in cleanup_paths:
            Path(path).unlink(missing_ok=True)
        raise

    extension = STREAM_FORMATS[output_format]["extension"]
    media_type = STREAM_FORMATS[output_format]["media_type"]

    def body() -> Iterator[bytes]:
        try:
            yield from encode_audio_stream(
                chunks,
                output_format,
                sample_rate,
                normalize_loudness=payload.normalize_loudness,
            )
            set_generation_activity("complete", "Stream is ready", active=False)
        except Exception as exc:
            set_generation_activity("failed", str(exc), active=False)
            raise
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()
            for path in cleanup_paths:
                Path(path).unlink(missing_ok=True)

    headers = {
        "Content-Disposition": f"inline; filename=voxcpm-stream.{extension}",
        "X-VoxCPM-Model": DEFAULT_MODEL_ID,
        "X-VoxCPM-Sample-Rate": str(sample_rate),
        "X-VoxCPM-Route": route_name,
        "X-VoxCPM-Format": output_format,
        "X-VoxCPM-Streaming": "progressive-chunks",
        "X-VoxCPM-Seed": str(seed),
        "X-VoxCPM-Input-Type": payload.input_type,
        "X-VoxCPM-Loudness-Normalized": str(payload.normalize_loudness).lower(),
        "X-VoxCPM-Long-Text-Protected": str(payload.protect_long_audio).lower(),
    }
    return StreamingResponse(body(), media_type=media_type, headers=headers)


def get_supported_output_formats() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "label": config["label"],
            "extension": config["extension"],
            "media_type": config["media_type"],
            "ui": config.get("ui", True),
        }
        for key, config in OUTPUT_FORMATS.items()
    }


def get_status_payload() -> dict:
    loaded_devices = [device for _, _, device in MODEL_CACHE]
    return {
        "msg": "pong",
        "type": "VoxCPMTTS",
        "version": APP_VERSION,
        "build_id": BUILD_ID,
        "build_date": BUILD_DATE,
        "revision": VCS_REF,
        "runtime": get_runtime_label(),
        "device": DEFAULT_DEVICE,
        "backend": DEFAULT_BACKEND,
        "model_id": DEFAULT_MODEL_ID,
        "load_denoiser": DEFAULT_LOAD_DENOISER,
        "load_asr": DEFAULT_LOAD_ASR,
        "asr_model_id": ASR_MODEL_ID,
        "asr_model_revision": ASR_MODEL_REVISION,
        "asr_device": ASR_DEVICE,
        "asr_loaded": ASR_MODEL is not None,
        "optimize": DEFAULT_OPTIMIZE,
        "languages": SUPPORTED_LANGUAGES,
        "loaded_model_devices": loaded_devices,
        "output_formats": get_supported_output_formats(),
        "stream_formats": STREAM_FORMATS,
        "post_processing": {
            "methods": sorted(POST_PROCESSING_METHODS),
            "presets": ["clean", "studio", "custom"],
            "output_format": "wav",
        },
        "input_types": ["text", "ssml", "ssml-h"],
        "ssml": ssml_capabilities(),
        "timestamps": {
            "available": timestamp_backend_available(),
            "backend": "stable-ts",
            "levels": ["segment", "word", "char"],
            "model": TIMESTAMP_MODEL,
            "device": TIMESTAMP_DEVICE,
        },
        "hardware": [
            {"label": label, "value": value}
            for label, value in get_hardware_choices()
        ],
    }


def voice_profile_payloads() -> list[dict[str, Any]]:
    return [
        {
            "id": name,
            "profile_type": profile["profile_type"],
            "description": profile["description"],
            "tags": profile["tags"],
            "language": profile["language"],
            "has_audio": bool(profile["audio_file"]),
            "has_design_audio": bool(profile["design_audio_file"]),
            "has_portrait": bool(profile["portrait_file"]),
            "has_transcript": bool(profile["ref_text"]),
            "has_control": bool(profile["control"]),
            "ref_text": profile["ref_text"],
            "control": profile["control"],
            "recipe": profile["recipe"],
            "created_at": profile["created_at"],
            "audio_url": f"/tts/voice-profiles/{name}/audio" if profile["audio_file"] else None,
            "design_audio_url": f"/tts/voice-profiles/{name}/design-audio" if profile["design_audio_file"] else None,
            "portrait_url": f"/tts/voice-profiles/{name}/portrait" if profile["portrait_file"] else None,
        }
        for name, profile in sorted(load_voice_profiles(VOICE_PROFILE_DIR).items())
    ]


def dialogue_script_payload(name: str, script: dict[str, Any], *, include_document: bool = False) -> dict[str, Any]:
    payload = {
        "id": name,
        "name": script["title"],
        "description": script["description"],
        "tags": script["tags"],
        "turns": script["turns"],
        "words": script["words"],
        "created_at": script["created_at"],
        "updated_at": script["updated_at"],
        "download_url": f"/tts/dialogue-scripts/{name}/download",
    }
    if include_document:
        _, _, path = resolve_dialogue_script(DIALOGUE_SCRIPT_DIR, name)
        payload["document"] = path.read_text(encoding="utf-8")
    return payload


def dialogue_script_payloads() -> list[dict[str, Any]]:
    scripts = load_dialogue_scripts(DIALOGUE_SCRIPT_DIR)
    return [dialogue_script_payload(name, script) for name, script in sorted(scripts.items())]


def validate_dialogue_script(document: str) -> tuple[int, int]:
    plan = compile_ssml(document, "ssml-h")
    speech = [unit for unit in plan.units if unit.kind == "speech"]
    words = sum(len(str(getattr(unit, "text", "") or "").split()) for unit in speech)
    return len(speech), words


async def save_reference_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "reference.wav").suffix.lower()
    if suffix not in REFERENCE_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported reference audio file type")

    total = 0
    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as output:
            path = output.name
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_REFERENCE_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Reference audio exceeds the upload limit")
                output.write(chunk)
        if suffix in REFERENCE_AUDIO_TRANSCODE_SUFFIXES:
            normalized_path = await asyncio.to_thread(normalize_reference_audio, path)
            original_path = path
            path = normalized_path
            Path(original_path).unlink(missing_ok=True)
    except Exception:
        if path:
            Path(path).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return path


async def read_magic_take_uploads(
    uploads: list[UploadFile],
    metadata: list[dict[str, Any]],
) -> dict[str, tuple[int, int, np.ndarray]]:
    if len(uploads) != len(metadata):
        for upload in uploads:
            await upload.close()
        raise HTTPException(status_code=400, detail="Magic take metadata does not match the uploaded audio.")
    if len(uploads) > MAX_MAGIC_TAKES:
        for upload in uploads:
            await upload.close()
        raise HTTPException(status_code=400, detail=f"Magic supports at most {MAX_MAGIC_TAKES} uploaded takes.")

    total = 0
    decoded: dict[str, tuple[int, int, np.ndarray]] = {}
    try:
        for upload, item in zip(uploads, metadata, strict=True):
            take_id = str(item.get("id") or "").strip()
            if not take_id or take_id in decoded:
                raise HTTPException(status_code=400, detail="Magic uploaded take IDs must be present and unique.")
            try:
                seed = int(item.get("seed", 0))
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Magic take '{take_id}' has an invalid seed.") from exc
            if not 0 <= seed <= MAX_RANDOM_SEED:
                raise HTTPException(status_code=400, detail=f"Magic take '{take_id}' seed is outside the 32-bit range.")
            data = bytearray()
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_MAGIC_TAKE_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Magic take audio exceeds the aggregate upload limit.")
                data.extend(chunk)
            try:
                sample_rate, waveform = await asyncio.to_thread(decode_pcm16_mono_wav, bytes(data))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Magic take '{take_id}': {exc}") from exc
            decoded[take_id] = (seed, sample_rate, waveform)
    finally:
        for upload in uploads:
            await upload.close()
    return decoded


def normalize_reference_audio(source_path: str) -> str:
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = output.name
    output.close()
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-c:a",
        "pcm_f32le",
        "-threads",
        "1",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
            raise ValueError("Reference audio conversion produced no audio.")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        Path(output_path).unlink(missing_ok=True)
        detail = exc.stderr.decode("utf-8", errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise HTTPException(status_code=400, detail=f"Unable to process reference audio: {detail}") from exc
    return output_path


async def save_portrait_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "portrait.webp").suffix.lower()
    if suffix not in PORTRAIT_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported voice portrait file type")

    total = 0
    header = b""
    path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as output:
            path = output.name
            while chunk := await upload.read(1024 * 1024):
                if len(header) < 12:
                    header = (header + chunk)[:12]
                total += len(chunk)
                if total > MAX_PORTRAIT_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Voice portrait exceeds the 5 MB upload limit")
                output.write(chunk)
        is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = header.startswith(b"\xff\xd8\xff")
        is_webp = header.startswith(b"RIFF") and header[8:12] == b"WEBP"
        valid_signature = is_png or is_jpeg or is_webp
        valid_suffix = (is_png and suffix == ".png") or (is_jpeg and suffix in {".jpg", ".jpeg"}) or (is_webp and suffix == ".webp")
        if not valid_signature or not valid_suffix:
            raise HTTPException(status_code=400, detail="Voice portrait contents do not match its image type")
        normalized_path = await asyncio.to_thread(normalize_portrait_image, path)
        Path(path).unlink(missing_ok=True)
        path = normalized_path
    except Exception:
        if path:
            Path(path).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return path


def normalize_portrait_image(source_path: str) -> str:
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".webp")
    output_path = output.name
    output.close()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-frames:v",
        "1",
        "-vf",
        "scale=100:100:force_original_aspect_ratio=increase,crop=100:100,format=bgra",
        "-c:v",
        "libwebp",
        "-lossless",
        "1",
        "-compression_level",
        "6",
        "-preset",
        "picture",
        "-threads",
        "1",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
            raise ValueError("Voice portrait conversion produced no image.")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        Path(output_path).unlink(missing_ok=True)
        detail = exc.stderr.decode("utf-8", errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise HTTPException(status_code=400, detail=f"Unable to process voice portrait: {detail}") from exc
    return output_path


def parse_uploaded_payload(payload: str, *, streaming: bool = False) -> TTSRequest:
    try:
        parsed = json.loads(payload)
        request_type = StreamingTTSRequest if streaming else TTSRequest
        return request_type.model_validate(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid generation payload: {exc}") from exc


async def uploaded_audio_response(
    payload: str,
    reference_audio: UploadFile,
    *,
    streaming: bool,
) -> StreamingResponse:
    request = parse_uploaded_payload(payload, streaming=streaming)
    reference_path = await save_reference_upload(reference_audio)
    request.ref_audio = reference_path
    request.reference_audio = None
    route_name = "/tts/stream-upload" if streaming else "/tts/generate-upload"
    if streaming:
        assert isinstance(request, StreamingTTSRequest)
        return await asyncio.to_thread(
            progressive_audio_response,
            request,
            route_name,
            cleanup_paths=(reference_path,),
        )
    try:
        return await asyncio.to_thread(stream_audio_response, request, route_name)
    finally:
        Path(reference_path).unlink(missing_ok=True)


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    yield
    GPU_MONITOR.close()
    await asyncio.to_thread(_close_all_models)
    await asyncio.to_thread(_close_asr_model)


api = FastAPI(
    title="VoxCPMTTS Service API",
    description="OpenAI-compatible and native HTTP APIs for Hangry Labs VoxCPMTTS.",
    version=APP_VERSION,
    openapi_url="/tts/openapi.json",
    docs_url="/tts/docs",
    redoc_url="/tts/redoc",
    lifespan=app_lifespan,
)


@api.exception_handler(OpenAIAPIError)
async def openai_api_error_handler(_: Request, exc: OpenAIAPIError) -> JSONResponse:
    return openai_error_response(
        exc.message,
        status_code=exc.status_code,
        error_type=exc.error_type,
        param=exc.param,
        code=exc.code,
        headers=exc.headers,
    )


@api.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    if not request.url.path.startswith("/v1"):
        return await request_validation_exception_handler(request, exc)

    first_error = exc.errors()[0] if exc.errors() else {}
    location = first_error.get("loc", ())
    param = str(location[-1]) if location and location[-1] != "body" else None
    message = str(first_error.get("msg") or "Invalid request body.")
    if param:
        message = f"Invalid value for '{param}': {message}."
    return openai_error_response(
        message,
        status_code=400,
        param=param,
        code="invalid_parameter",
    )


@api.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> Response:
    if not request.url.path.startswith("/v1"):
        return await http_exception_handler(request, exc)
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    server_error = exc.status_code >= 500
    return openai_error_response(
        detail,
        status_code=exc.status_code,
        error_type="server_error" if server_error else "invalid_request_error",
        code="internal_server_error" if server_error else ("not_found" if exc.status_code == 404 else None),
        headers=exc.headers,
    )


OPENAI_API_DEPENDENCIES = [Depends(require_openai_api_key)]


@api.get("/v1", tags=["OpenAI compatibility"], dependencies=OPENAI_API_DEPENDENCIES)
@api.get("/v1/", tags=["OpenAI compatibility"], dependencies=OPENAI_API_DEPENDENCIES)
def openai_api_index() -> dict[str, Any]:
    return {
        "object": "api",
        "name": "VoxCPMTTS OpenAI-compatible speech API",
        "version": APP_VERSION,
        "model": OPENAI_MODEL_ID,
        "endpoints": ["/v1/models", "/v1/audio/voices", "/v1/audio/speech"],
    }


@api.get("/v1/models", tags=["OpenAI compatibility"], dependencies=OPENAI_API_DEPENDENCIES)
def openai_models() -> dict[str, Any]:
    return {"object": "list", "data": [openai_model_payload()]}


@api.get("/v1/models/{model_id:path}", tags=["OpenAI compatibility"], dependencies=OPENAI_API_DEPENDENCIES)
def openai_model(model_id: str) -> dict[str, Any]:
    resolve_openai_model(model_id)
    return openai_model_payload(model_id)


@api.get("/v1/audio/voices", tags=["OpenAI compatibility"], dependencies=OPENAI_API_DEPENDENCIES)
def openai_voices() -> dict[str, Any]:
    voices = []
    for voice_id, (description, _) in OPENAI_VOICE_PRESETS.items():
        voice = openai_voice_payload(voice_id, profile_type="preset", owned_by="hangry-labs")
        voice["description"] = description
        voices.append(voice)
    for voice_id, profile in sorted(load_voice_profiles(VOICE_PROFILE_DIR).items()):
        voice = openai_voice_payload(voice_id, profile_type=profile["profile_type"], owned_by="local")
        voice.update(
            {
                "description": profile["description"],
                "language": profile["language"],
                "tags": profile["tags"],
            }
        )
        voices.append(voice)
    return {"object": "list", "data": voices}


@api.post("/v1/audio/speech", tags=["OpenAI compatibility"], dependencies=OPENAI_API_DEPENDENCIES)
def openai_speech(payload: OpenAISpeechRequest) -> StreamingResponse:
    return stream_audio_response(openai_speech_to_tts_request(payload), "/v1/audio/speech")


@api.get("/tts/ping")
def ping() -> dict:
    return {
        "msg": "pong",
        "type": "VoxCPMTTS",
        "version": APP_VERSION,
        "build_id": BUILD_ID,
        "build_date": BUILD_DATE,
        "revision": VCS_REF,
        "backend": DEFAULT_BACKEND,
    }


@api.get("/tts/status")
def status() -> dict:
    return get_status_payload()


@api.get("/tts/activity")
def activity() -> dict:
    return get_generation_activity()


@api.get("/tts/defaults")
def defaults() -> dict:
    return {
        "text": "Hello from Hangry Labs VoxCPMTTS.",
        "language": "English",
        "voice": "auto",
        "device": "auto",
        "cfg_value": 2.0,
        "inference_timesteps": DEFAULT_INFERENCE_TIMESTEPS,
        "seed": 42,
        "randomize_seed": True,
        "normalize_loudness": True,
        "protect_long_audio": True,
        "output_formats": {"default": "wav", "available": get_supported_output_formats()},
        "stream_formats": {"default": "mp3", "available": STREAM_FORMATS},
    }


@api.get("/tts/formats")
def formats() -> dict:
    return {"default": "wav", "formats": get_supported_output_formats(), "aliases": FORMAT_ALIASES}


@api.get("/tts/stream-formats")
def stream_formats() -> dict:
    return {
        "default": "mp3",
        "formats": STREAM_FORMATS,
        "aliases": STREAM_FORMAT_ALIASES,
        "granularity": "progressive_model_chunks",
    }


@api.get("/tts/languages")
def languages() -> dict:
    return {"languages": SUPPORTED_LANGUAGES, "dialect_note": "Chinese dialect generation is text/control driven."}


@api.get("/tts/voices")
def voices() -> dict:
    saved = voice_profile_payloads()
    return {
        "voices": [
            {"id": "auto", "name": "Auto Voice Design", "language": "auto"},
            {"id": "reference", "name": "Reference Audio Clone", "language": "auto"},
        ],
        "saved_profiles": saved,
        "note": "VoxCPM2 does not use a fixed speaker inventory. Use control/instruct or ref_audio.",
    }


@api.get("/tts/voice-profiles")
def voice_profiles(response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    profiles = voice_profile_payloads()
    return {"object": "list", "data": profiles, "count": len(profiles)}


@api.post("/tts/voice-profiles")
async def create_voice_profile(
    name: str = Form(...),
    profile_type: str = Form(...),
    description: str = Form(""),
    tags: str = Form(""),
    ref_text: str = Form(""),
    control: str = Form(""),
    language: str = Form(""),
    recipe: str = Form(""),
    reference_audio: UploadFile | None = File(None),
    design_reference_audio: UploadFile | None = File(None),
    portrait: UploadFile | None = File(None),
) -> dict:
    temporary_path: str | None = None
    design_temporary_path: str | None = None
    portrait_temporary_path: str | None = None
    try:
        if reference_audio is not None:
            temporary_path = await save_reference_upload(reference_audio)
        if design_reference_audio is not None:
            design_temporary_path = await save_reference_upload(design_reference_audio)
        if portrait is not None:
            portrait_temporary_path = await save_portrait_upload(portrait)
        parsed_recipe = normalize_profile_recipe(json.loads(recipe)) if recipe.strip() else {}
        parsed_tags = normalize_profile_tags(json.loads(tags)) if tags.strip() else []
        profile_name, _ = await asyncio.to_thread(
            save_voice_profile,
            VOICE_PROFILE_DIR,
            name=name,
            profile_type=profile_type,
            source_audio=temporary_path,
            design_source_audio=design_temporary_path,
            portrait_source=portrait_temporary_path,
            ref_text=ref_text,
            control=control,
            description=description,
            tags=parsed_tags,
            language=language,
            recipe=parsed_recipe,
            overwrite=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
        if design_temporary_path:
            Path(design_temporary_path).unlink(missing_ok=True)
        if portrait_temporary_path:
            Path(portrait_temporary_path).unlink(missing_ok=True)
    return next(profile for profile in voice_profile_payloads() if profile["id"] == profile_name)


@api.put("/tts/voice-profiles/{profile_name}")
async def edit_voice_profile(
    profile_name: str,
    profile_type: str | None = Form(None),
    description: str | None = Form(None),
    tags: str = Form(""),
    ref_text: str | None = Form(None),
    control: str | None = Form(None),
    language: str | None = Form(None),
    recipe: str = Form(""),
    clear_design_audio: bool = Form(False),
    clear_portrait: bool = Form(False),
    reference_audio: UploadFile | None = File(None),
    design_reference_audio: UploadFile | None = File(None),
    portrait: UploadFile | None = File(None),
) -> dict:
    temporary_path: str | None = None
    design_temporary_path: str | None = None
    portrait_temporary_path: str | None = None
    try:
        if reference_audio is not None:
            temporary_path = await save_reference_upload(reference_audio)
        if design_reference_audio is not None:
            design_temporary_path = await save_reference_upload(design_reference_audio)
        if portrait is not None:
            portrait_temporary_path = await save_portrait_upload(portrait)
        parsed_recipe = normalize_profile_recipe(json.loads(recipe)) if recipe.strip() else None
        parsed_tags = normalize_profile_tags(json.loads(tags)) if tags.strip() else None
        normalized_name, _ = await asyncio.to_thread(
            update_voice_profile,
            VOICE_PROFILE_DIR,
            name=profile_name,
            source_audio=temporary_path,
            design_source_audio=design_temporary_path,
            clear_design_audio=clear_design_audio,
            portrait_source=portrait_temporary_path,
            clear_portrait=clear_portrait,
            profile_type=profile_type,
            ref_text=ref_text,
            control=control,
            description=description,
            tags=parsed_tags,
            language=language,
            recipe=parsed_recipe,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
        if design_temporary_path:
            Path(design_temporary_path).unlink(missing_ok=True)
        if portrait_temporary_path:
            Path(portrait_temporary_path).unlink(missing_ok=True)
    return next(profile for profile in voice_profile_payloads() if profile["id"] == normalized_name)


@api.get("/tts/voice-profiles/{profile_name}/audio")
def voice_profile_audio(profile_name: str) -> FileResponse:
    try:
        _, profile = resolve_voice_profile(VOICE_PROFILE_DIR, profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audio_path = profile.get("ref_audio")
    if not audio_path:
        raise HTTPException(status_code=404, detail="Designed voice profiles do not contain reference audio.")
    return FileResponse(audio_path, filename=Path(audio_path).name)


@api.get("/tts/voice-profiles/{profile_name}/design-audio")
def voice_profile_design_audio(profile_name: str) -> FileResponse:
    try:
        _, profile = resolve_voice_profile(VOICE_PROFILE_DIR, profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audio_path = profile.get("design_ref_audio")
    if not audio_path:
        raise HTTPException(status_code=404, detail="Voice profile has no stored design reference audio.")
    return FileResponse(audio_path, filename=Path(audio_path).name)


@api.get("/tts/voice-profiles/{profile_name}/portrait")
def voice_profile_portrait(profile_name: str) -> FileResponse:
    try:
        _, profile = resolve_voice_profile(VOICE_PROFILE_DIR, profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    portrait_path = profile.get("portrait_path")
    if not portrait_path:
        raise HTTPException(status_code=404, detail="Voice profile has no portrait.")
    return FileResponse(portrait_path, headers={"Cache-Control": "no-store"})


@api.delete("/tts/voice-profiles/{profile_name}")
def remove_voice_profile(profile_name: str) -> dict[str, str]:
    try:
        deleted = delete_voice_profile(VOICE_PROFILE_DIR, profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": deleted}


@api.get("/tts/dialogue-scripts")
def dialogue_scripts(response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    scripts = dialogue_script_payloads()
    return {"object": "list", "data": scripts, "count": len(scripts)}


@api.post("/tts/dialogue-scripts")
def create_dialogue_script(payload: DialogueScriptRequest) -> dict:
    try:
        turns, words = validate_dialogue_script(payload.document)
        script_name, script = save_dialogue_script(
            DIALOGUE_SCRIPT_DIR,
            name=payload.name,
            document=payload.document,
            description=payload.description,
            tags=payload.tags,
            turns=turns,
            words=words,
            overwrite=False,
        )
    except (SSMLValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dialogue_script_payload(script_name, script, include_document=True)


@api.get("/tts/dialogue-scripts/{script_name}")
def get_dialogue_script(script_name: str) -> dict:
    try:
        name, script, _ = resolve_dialogue_script(DIALOGUE_SCRIPT_DIR, script_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return dialogue_script_payload(name, script, include_document=True)


@api.put("/tts/dialogue-scripts/{script_name}")
def edit_dialogue_script(script_name: str, payload: DialogueScriptRequest) -> dict:
    try:
        normalized_name, _, _ = resolve_dialogue_script(DIALOGUE_SCRIPT_DIR, script_name)
        if normalize_script_name(payload.name) != normalized_name:
            raise ValueError("Create a new script to use a different name.")
        turns, words = validate_dialogue_script(payload.document)
        name, script = save_dialogue_script(
            DIALOGUE_SCRIPT_DIR,
            name=payload.name,
            document=payload.document,
            description=payload.description,
            tags=payload.tags,
            turns=turns,
            words=words,
            overwrite=True,
        )
    except (SSMLValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dialogue_script_payload(name, script, include_document=True)


@api.get("/tts/dialogue-scripts/{script_name}/download")
def download_dialogue_script(script_name: str) -> FileResponse:
    try:
        name, _, path = resolve_dialogue_script(DIALOGUE_SCRIPT_DIR, script_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=f"{name}.ssml", media_type="application/ssml+xml")


@api.delete("/tts/dialogue-scripts/{script_name}")
def remove_dialogue_script(script_name: str) -> dict[str, str]:
    try:
        deleted = delete_dialogue_script(DIALOGUE_SCRIPT_DIR, script_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": deleted}


@api.get("/tts/speakers")
def speakers(language: str = Query("auto", description="Compatibility parameter.")) -> dict:
    return {"language": language, "speakers": ["auto", "reference"]}


@api.post("/tts/metrics")
def metrics(payload: MetricsRequest = Body(...)) -> dict:
    text = payload.text or ""
    metrics_payload: dict[str, Any] = {"characters": len(text), "words": len(text.split())}
    if payload.input_type != "text":
        request = TTSRequest(text=text, input_type=payload.input_type, language=payload.language)
        plan, _ = compile_ssml_request(request)
        metrics_payload.update(
            units=len(plan.units),
            speech_units=sum(unit.kind == "speech" for unit in plan.units),
            break_ms=sum(unit.duration_ms for unit in plan.units if unit.kind == "break"),
            voices=list(plan.voices),
            languages=list(plan.languages),
            voice_definitions=len(plan.voice_definitions),
        )
    return {"input_type": payload.input_type, "metrics": metrics_payload}


@api.get("/tts/ssml/capabilities")
def get_ssml_capabilities() -> dict:
    return ssml_capabilities()


@api.post("/tts/transcribe")
def transcribe(payload: TranscriptionRequest = Body(...)) -> dict:
    try:
        text = transcribe_reference_audio(payload.audio_path, payload.language)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reference transcription failed: {exc}") from exc
    return {
        "text": text,
        "language": payload.language,
        "model_id": ASR_MODEL_ID,
        "model_revision": ASR_MODEL_REVISION,
    }


@api.post("/tts/transcribe-upload")
async def transcribe_upload(
    reference_audio: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict:
    reference_path = await save_reference_upload(reference_audio)
    try:
        text = await asyncio.to_thread(transcribe_reference_audio, reference_path, language)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reference transcription failed: {exc}") from exc
    finally:
        Path(reference_path).unlink(missing_ok=True)
    return {
        "text": text,
        "language": language,
        "model_id": ASR_MODEL_ID,
        "model_revision": ASR_MODEL_REVISION,
    }


@api.post("/tts/timestamps-upload")
async def timestamps_upload(
    audio: UploadFile = File(...),
    text: str = Form(...),
    level: str = Form("word"),
    language: str = Form(""),
) -> dict:
    if level not in {"segment", "word", "char"}:
        raise HTTPException(status_code=400, detail="Timestamp level must be segment, word, or char")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Timestamp text must not be empty")
    if not timestamp_backend_available():
        raise HTTPException(
            status_code=503,
            detail="Timestamp alignment is unavailable because the stable-ts backend is not installed.",
        )

    audio_path = await save_reference_upload(audio)
    try:
        return await asyncio.to_thread(
            align_audio_file,
            audio_path=audio_path,
            text=text.strip(),
            backend="stable-ts",
            level=level,
            model_name=TIMESTAMP_MODEL,
            device=TIMESTAMP_DEVICE,
            language=language.strip() or None,
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Timestamp alignment failed: {exc}") from exc
    finally:
        Path(audio_path).unlink(missing_ok=True)


@api.post("/tts/postprocess-upload")
async def postprocess_upload(
    audio: UploadFile = File(...),
    options: str = Form(...),
) -> Response:
    try:
        payload = PostProcessingRequest.model_validate_json(options)
    except ValueError as exc:
        await audio.close()
        raise HTTPException(status_code=422, detail=f"Invalid post-processing options: {exc}") from exc

    audio_path = await save_reference_upload(audio)
    try:
        processed = await asyncio.to_thread(post_process_audio, audio_path, payload)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(audio_path).unlink(missing_ok=True)
    return Response(
        content=processed,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="voxcpmtts-processed.wav"',
            "X-VoxCPM-Post-Processing": payload.method,
            "X-VoxCPM-Post-Processing-Preset": payload.preset,
        },
    )


@api.post("/tts/magic/take")
def generate_magic_take(request: MagicTakeRequest = Body(...)) -> Response:
    try:
        set_generation_activity("generating", "Generating Magic speech take", active=True)
        audio, seed, sample_rate = render_magic_take(request.payload, request.speech_index)
        set_generation_activity("complete", "Magic speech take is ready", active=False)
    except (FileNotFoundError, ValueError, RuntimeError, SSMLValidationError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"Magic take generation failed: {exc}") from exc
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="voxcpmtts-take-{request.speech_index + 1}.wav"',
            "X-VoxCPM-Seed": str(seed),
            "X-VoxCPM-Sample-Rate": str(sample_rate),
            "X-VoxCPM-Format": "wav",
            "X-VoxCPM-Route": "/tts/magic/take",
        },
    )


@api.post("/tts/magic/render")
async def generate_magic_render(
    payload: str = Form(...),
    speech_ids: str = Form(...),
    retained: str = Form("[]"),
    takes: list[UploadFile] = File(default=[]),
) -> Response:
    try:
        request = TTSRequest.model_validate_json(payload)
        parsed_ids = json.loads(speech_ids)
        parsed_retained = json.loads(retained)
        if not isinstance(parsed_ids, list) or not all(isinstance(item, str) for item in parsed_ids):
            raise ValueError("Magic speech_ids must be a JSON array of strings.")
        if not isinstance(parsed_retained, list) or not all(isinstance(item, dict) for item in parsed_retained):
            raise ValueError("Magic retained metadata must be a JSON array of objects.")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        for upload in takes:
            await upload.close()
        raise HTTPException(status_code=422, detail=f"Invalid Magic render request: {exc}") from exc

    retained_takes = await read_magic_take_uploads(takes, parsed_retained)
    try:
        set_generation_activity("generating", "Generating unlocked Magic speech takes", active=True)
        bundle, seed, output_format, sample_rate = await asyncio.to_thread(
            render_magic_take_bundle,
            request,
            parsed_ids,
            retained_takes,
        )
        set_generation_activity("complete", "Magic dialogue is ready", active=False)
    except (FileNotFoundError, ValueError, RuntimeError, SSMLValidationError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"Magic dialogue generation failed: {exc}") from exc
    return Response(
        content=bundle,
        media_type=MAGIC_TAKE_MEDIA_TYPE,
        headers={
            "Content-Disposition": 'inline; filename="voxcpmtts-magic-takes.bin"',
            "X-VoxCPM-Seed": str(seed),
            "X-VoxCPM-Sample-Rate": str(sample_rate),
            "X-VoxCPM-Format": output_format,
            "X-VoxCPM-Route": "/tts/magic/render",
        },
    )


@api.post("/tts/magic/assemble-upload")
async def assemble_magic_takes(
    options: str = Form(...),
    take_metadata: str = Form(...),
    takes: list[UploadFile] = File(...),
) -> Response:
    try:
        request = MagicAssemblyRequest.model_validate_json(options)
        metadata = json.loads(take_metadata)
        if not isinstance(metadata, list) or not all(isinstance(item, dict) for item in metadata):
            raise ValueError("Magic take metadata must be a JSON array of objects.")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        for upload in takes:
            await upload.close()
        raise HTTPException(status_code=422, detail=f"Invalid Magic assembly request: {exc}") from exc

    decoded = await read_magic_take_uploads(takes, metadata)
    assembly_takes = {take_id: (sample_rate, waveform) for take_id, (_, sample_rate, waveform) in decoded.items()}
    try:
        audio, output_format, sample_rate = await asyncio.to_thread(
            assemble_magic_uploads,
            request,
            assembly_takes,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    extension = OUTPUT_FORMATS[output_format]["extension"]
    return Response(
        content=audio,
        media_type=OUTPUT_FORMATS[output_format]["media_type"],
        headers={
            "Content-Disposition": f'inline; filename="voxcpmtts-magic-preview.{extension}"',
            "X-VoxCPM-Sample-Rate": str(OUTPUT_FORMATS[output_format].get("sample_rate", sample_rate)),
            "X-VoxCPM-Format": output_format,
            "X-VoxCPM-Route": "/tts/magic/assemble-upload",
        },
    )


@api.post("/tts/generate")
def generate_tts(payload: TTSRequest = Body(...)) -> StreamingResponse:
    return stream_audio_response(payload, "/tts/generate")


@api.post("/tts/generate-upload")
async def generate_tts_upload(
    payload: str = Form(...),
    reference_audio: UploadFile = File(...),
) -> StreamingResponse:
    return await uploaded_audio_response(payload, reference_audio, streaming=False)


@api.post("/tts/convert")
def convert(payload: TTSRequest = Body(...)) -> StreamingResponse:
    return stream_audio_response(payload, "/tts/convert")


@api.post("/tts/stream")
def stream_tts(payload: StreamingTTSRequest = Body(...)) -> StreamingResponse:
    return progressive_audio_response(payload, "/tts/stream")


@api.post("/tts/stream-upload")
async def stream_tts_upload(
    payload: str = Form(...),
    reference_audio: UploadFile = File(...),
) -> StreamingResponse:
    return await uploaded_audio_response(payload, reference_audio, streaming=True)


@api.post("/tts/purge")
def purge_models(payload: PurgeRequest | None = Body(None)) -> dict:
    requested_device = payload.device if payload else None
    with MODEL_LOCK:
        if requested_device:
            device = canonical_model_device(resolve_requested_device(requested_device))
            purged = []
            for cache_key in list(MODEL_CACHE):
                if cache_key[2] == device:
                    _close_model(MODEL_CACHE.pop(cache_key))
                    purged.append(device)
            remaining = [cached_device for _, _, cached_device in MODEL_CACHE]
            return {"purged": purged, "remaining_model_devices": remaining}

        purged = [device for _, _, device in MODEL_CACHE]
        for model in MODEL_CACHE.values():
            _close_model(model)
        MODEL_CACHE.clear()
        if _close_asr_model():
            purged.append("asr")
        return {"purged": purged, "remaining_model_devices": []}


app = attach_ui(api_app=api)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8808"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run("voxcpm.app:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
