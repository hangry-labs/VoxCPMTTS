from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from voxcpm.standalone_ui.gpu import GPU_MONITOR
from voxcpm.standalone_ui.server import attach_ui
from voxcpm.voice_profiles import (
    delete_voice_profile,
    load_voice_profiles,
    resolve_voice_profile,
    save_voice_profile,
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
REFERENCE_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
VOICE_PROFILE_DIR = Path(os.getenv("VOXCPM_VOICE_PROFILE_DIR", "/app/persistent/voices"))

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
}
FORMAT_ALIASES = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".ogg": "ogg",
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
ASR_MODEL_ID = os.getenv("VOXCPM_ASR_MODEL_ID", "iic/SenseVoiceSmall")
DEFAULT_LOAD_ASR = os.getenv("VOXCPM_LOAD_ASR", "1").lower() in {"1", "true", "yes"}

MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
MODEL_LOCK = threading.Lock()
ASR_MODEL = None
ASR_LOCK = threading.Lock()
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


def get_asr_model():
    global ASR_MODEL
    if not DEFAULT_LOAD_ASR:
        raise RuntimeError("ASR is disabled. Set VOXCPM_LOAD_ASR=1 to enable reference transcription.")
    with ASR_LOCK:
        if ASR_MODEL is None:
            from funasr import AutoModel

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            ASR_MODEL = AutoModel(
                model=ASR_MODEL_ID,
                disable_update=True,
                log_level="ERROR",
                device=device,
            )
        return ASR_MODEL


def transcribe_reference_audio(audio_path: str, language: str = "auto") -> str:
    if not audio_path:
        raise ValueError("audio_path is required")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_asr_model()
    result = model.generate(input=audio_path, language=language or "auto", use_itn=True)
    if not result:
        return ""
    text = result[0].get("text", "")
    return text.split("|>")[-1].strip()


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


def build_generate_kwargs(payload: "TTSRequest", model: Any) -> dict:
    ref_audio = payload.ref_audio or payload.reference_audio
    prompt_audio = payload.prompt_audio
    prompt_text = payload.prompt_text
    control = payload.control or payload.instruct
    if payload.voice_profile:
        _, profile = resolve_voice_profile(VOICE_PROFILE_DIR, payload.voice_profile)
        if not ref_audio and not prompt_audio:
            ref_audio = profile.get("ref_audio") or None
        if not prompt_text and not payload.ref_text:
            prompt_text = profile.get("ref_text") or None
        if not control:
            control = profile.get("control") or None
    if payload.ref_text and not prompt_text:
        prompt_text = payload.ref_text
    if ref_audio and prompt_text and not prompt_audio:
        prompt_audio = ref_audio

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
    }
    if not prompt_audio:
        kwargs["prompt_text"] = None
    if not ref_audio:
        kwargs["reference_wav_path"] = None
    return kwargs


class TTSRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1, description="Text to synthesize.")
    language: str = Field("English", description="Compatibility hint; VoxCPM2 auto-detects supported languages.")
    voice: str = Field("auto", description="Compatibility field. Use instruct/control or reference audio for VoxCPM voices.")
    voice_profile: Optional[str] = Field(None, description="Saved local voice profile name.")
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
    denoise: bool = Field(False, description="Apply ZipEnhancer to prompt/reference audio when denoiser is enabled.")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Compatibility field; VoxCPM2 has no direct speed scalar.")
    device: str = Field("auto", description="auto, cpu, mps, cuda, or cuda:N.")
    use_gpu: Optional[bool] = Field(None, description="Legacy compatibility switch. Prefer device.")
    output_format: str = Field(
        "wav",
        alias="format",
        description="Response audio format. Supported: wav, mp3, flac, ogg.",
    )


class StreamingTTSRequest(TTSRequest):
    stream_format: str = Field("mp3", description="Progressive response format. Supported: mp3.")


class MetricsRequest(BaseModel):
    text: str = Field("", description="Text to inspect.")


class PurgeRequest(BaseModel):
    device: Optional[str] = Field(None, description="Optional cached device to clear. Omit to clear all cached models.")


class TranscriptionRequest(BaseModel):
    audio_path: str = Field(..., description="Container-visible audio path to transcribe.")
    language: str = Field("auto", description="ASR language hint. Use auto for detection.")


def synthesize_payload(payload: TTSRequest) -> tuple[str, int, np.ndarray]:
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
        wav = model.generate(**build_generate_kwargs(payload, model))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"VoxCPM generation failed: {exc}") from exc

    return output_format, int(model.tts_model.sample_rate), to_float32_audio(wav)


def synthesize_payload_chunks(payload: StreamingTTSRequest) -> tuple[str, int, Iterator[np.ndarray]]:
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
        chunks = model.generate_streaming(**build_generate_kwargs(payload, model))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        set_generation_activity("failed", str(exc), active=False)
        raise HTTPException(status_code=500, detail=f"VoxCPM streaming failed: {exc}") from exc

    return output_format, int(model.tts_model.sample_rate), chunks


def stream_audio_response(payload: TTSRequest, route_name: str) -> StreamingResponse:
    output_format, sample_rate, waveform = synthesize_payload(payload)
    try:
        set_generation_activity("encoding", f"Encoding {output_format.upper()} audio", active=True)
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
        "X-VoxCPM-Sample-Rate": str(sample_rate),
        "X-VoxCPM-Duration": f"{duration:.3f}",
        "X-VoxCPM-Route": route_name,
        "X-VoxCPM-Format": output_format,
    }
    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type, headers=headers)


def progressive_audio_response(
    payload: StreamingTTSRequest,
    route_name: str,
    *,
    cleanup_paths: tuple[str, ...] = (),
) -> StreamingResponse:
    try:
        output_format, sample_rate, chunks = synthesize_payload_chunks(payload)
    except Exception:
        for path in cleanup_paths:
            Path(path).unlink(missing_ok=True)
        raise

    extension = STREAM_FORMATS[output_format]["extension"]
    media_type = STREAM_FORMATS[output_format]["media_type"]

    def body() -> Iterator[bytes]:
        try:
            yield from encode_audio_stream(chunks, output_format, sample_rate)
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
    }
    return StreamingResponse(body(), media_type=media_type, headers=headers)


def get_supported_output_formats() -> dict[str, dict[str, str]]:
    return {
        key: {
            "label": config["label"],
            "extension": config["extension"],
            "media_type": config["media_type"],
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
        "optimize": DEFAULT_OPTIMIZE,
        "languages": SUPPORTED_LANGUAGES,
        "loaded_model_devices": loaded_devices,
        "output_formats": get_supported_output_formats(),
        "stream_formats": STREAM_FORMATS,
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
            "language": profile["language"],
            "has_audio": bool(profile["audio_file"]),
            "has_transcript": bool(profile["ref_text"]),
            "has_control": bool(profile["control"]),
            "created_at": profile["created_at"],
            "audio_url": f"/tts/voice-profiles/{name}/audio" if profile["audio_file"] else None,
        }
        for name, profile in sorted(load_voice_profiles(VOICE_PROFILE_DIR).items())
    ]


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
    except Exception:
        if path:
            Path(path).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return path


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


api = FastAPI(
    title="VoxCPMTTS Service API",
    description="HTTP API for Hangry Labs VoxCPMTTS.",
    version=APP_VERSION,
    openapi_url="/tts/openapi.json",
    docs_url="/tts/docs",
    redoc_url="/tts/redoc",
    lifespan=app_lifespan,
)


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
def voice_profiles() -> dict:
    profiles = voice_profile_payloads()
    return {"object": "list", "data": profiles, "count": len(profiles)}


@api.post("/tts/voice-profiles")
async def create_voice_profile(
    name: str = Form(...),
    profile_type: str = Form(...),
    description: str = Form(""),
    ref_text: str = Form(""),
    control: str = Form(""),
    language: str = Form(""),
    reference_audio: UploadFile | None = File(None),
) -> dict:
    temporary_path: str | None = None
    try:
        if reference_audio is not None:
            temporary_path = await save_reference_upload(reference_audio)
        profile_name, _ = await asyncio.to_thread(
            save_voice_profile,
            VOICE_PROFILE_DIR,
            name=name,
            profile_type=profile_type,
            source_audio=temporary_path,
            ref_text=ref_text,
            control=control,
            description=description,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
    return next(profile for profile in voice_profile_payloads() if profile["id"] == profile_name)


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


@api.delete("/tts/voice-profiles/{profile_name}")
def remove_voice_profile(profile_name: str) -> dict[str, str]:
    try:
        deleted = delete_voice_profile(VOICE_PROFILE_DIR, profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": deleted}


@api.get("/tts/speakers")
def speakers(language: str = Query("auto", description="Compatibility parameter.")) -> dict:
    return {"language": language, "speakers": ["auto", "reference"]}


@api.post("/tts/metrics")
def metrics(payload: MetricsRequest = Body(...)) -> dict:
    text = payload.text or ""
    return {"metrics": {"characters": len(text), "words": len(text.split())}}


@api.post("/tts/transcribe")
def transcribe(payload: TranscriptionRequest = Body(...)) -> dict:
    try:
        text = transcribe_reference_audio(payload.audio_path, payload.language)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reference transcription failed: {exc}") from exc
    return {"text": text, "language": payload.language, "model_id": ASR_MODEL_ID}


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
    return {"text": text, "language": language, "model_id": ASR_MODEL_ID}


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
        return {"purged": purged, "remaining_model_devices": []}


app = attach_ui(api_app=api)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8808"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run("voxcpm.app:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
