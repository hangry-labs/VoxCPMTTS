from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from voxcpm import VoxCPM


def _read_version() -> str:
    version_path = Path(__file__).resolve().parents[1] / "VERSION"
    if version_path.exists():
        return version_path.read_text(encoding="utf-8").strip()
    return "0.1-snapshot"


DEFAULT_MODEL_ID = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
DEFAULT_DEVICE = os.getenv("VOXCPM_DEVICE", "auto")
DEFAULT_LOAD_DENOISER = os.getenv("VOXCPM_LOAD_DENOISER", "0").lower() in {"1", "true", "yes"}
DEFAULT_OPTIMIZE = os.getenv("VOXCPM_OPTIMIZE", "1").lower() not in {"0", "false", "no"}
DEFAULT_LOCAL_ONLY = os.getenv("VOXCPM_LOCAL_FILES_ONLY", "0").lower() in {"1", "true", "yes"}
ZIPENHANCER_MODEL_ID = os.getenv("ZIPENHANCER_MODEL_ID", "iic/speech_zipenhancer_ans_multiloss_16k_base")
APP_VERSION = os.getenv("APP_VERSION", _read_version())
BUILD_ID = os.getenv("BUILD_ID", "stable")

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
    "wav": {
        "label": "Full WAV response",
        "extension": "wav",
        "media_type": "audio/wav",
    },
    "mp3": {
        "label": "Full MP3 response",
        "extension": "mp3",
        "media_type": "audio/mpeg",
    },
}
STREAM_FORMAT_ALIASES = {
    ".wav": "wav",
    ".mp3": "mp3",
    "mpeg": "mp3",
}
ASR_MODEL_ID = os.getenv("VOXCPM_ASR_MODEL_ID", "iic/SenseVoiceSmall")
DEFAULT_LOAD_ASR = os.getenv("VOXCPM_LOAD_ASR", "1").lower() in {"1", "true", "yes"}

MODEL_CACHE: dict[tuple[str, str], VoxCPM] = {}
MODEL_LOCK = threading.Lock()
ASR_MODEL = None
ASR_LOCK = threading.Lock()


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


def get_model(device: str = DEFAULT_DEVICE) -> VoxCPM:
    requested_device = resolve_requested_device(device)
    cache_key = (DEFAULT_MODEL_ID, requested_device)
    with MODEL_LOCK:
        if cache_key not in MODEL_CACHE:
            MODEL_CACHE[cache_key] = VoxCPM.from_pretrained(
                hf_model_id=DEFAULT_MODEL_ID,
                load_denoiser=DEFAULT_LOAD_DENOISER,
                zipenhancer_model_id=ZIPENHANCER_MODEL_ID,
                local_files_only=DEFAULT_LOCAL_ONLY,
                optimize=DEFAULT_OPTIMIZE,
                device=requested_device,
            )
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
    normalized = (stream_format or "wav").strip().lower()
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


def encoded_audio_to_temp_file(audio: np.ndarray, output_format: str, sample_rate: int) -> str:
    normalized_format = normalize_output_format(output_format)
    extension = OUTPUT_FORMATS[normalized_format]["extension"]
    audio_bytes = encode_audio_bytes(audio, normalized_format, sample_rate)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as file:
        file.write(audio_bytes)
        return file.name


def clean_control(control: str | None) -> str:
    control = (control or "").strip()
    return control.replace("(", "").replace(")", "").replace("（", "").replace("）", "").strip()


def build_final_text(text: str, control: str | None, prompt_text: str | None) -> str:
    stripped = (text or "").strip()
    control = clean_control(control)
    if control and not prompt_text:
        return f"({control}){stripped}"
    return stripped


def build_generate_kwargs(payload: "TTSRequest", model: VoxCPM) -> dict:
    ref_audio = payload.ref_audio or payload.reference_audio
    prompt_audio = payload.prompt_audio
    prompt_text = payload.prompt_text
    if payload.ref_text and not prompt_text:
        prompt_text = payload.ref_text
    if ref_audio and prompt_text and not prompt_audio:
        prompt_audio = ref_audio

    final_text = build_final_text(payload.text, payload.control or payload.instruct, prompt_text)
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
    control: Optional[str] = Field(None, description="VoxCPM voice design/control instruction.")
    instruct: Optional[str] = Field(None, description="Alias for control.")
    reference_audio: Optional[str] = Field(None, description="Container-visible reference audio path.")
    ref_audio: Optional[str] = Field(None, description="Alias for reference_audio.")
    ref_text: Optional[str] = Field(None, description="Transcript of the reference audio for ultimate cloning.")
    prompt_audio: Optional[str] = Field(None, description="Prompt audio path for continuation cloning.")
    prompt_text: Optional[str] = Field(None, description="Transcript for prompt_audio.")
    cfg_value: float = Field(2.0, ge=0.1, le=10.0, description="Classifier-free guidance scale.")
    inference_timesteps: int = Field(10, ge=1, le=100, description="LocDiT flow-matching steps.")
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
    stream_format: str = Field("wav", description="Streaming response format. Supported: wav, mp3.")


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
        model = get_model(requested_device)
        wav = model.generate(**build_generate_kwargs(payload, model))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VoxCPM generation failed: {exc}") from exc

    return output_format, int(model.tts_model.sample_rate), to_float32_audio(wav)


def stream_audio_response(payload: TTSRequest, route_name: str) -> StreamingResponse:
    output_format, sample_rate, waveform = synthesize_payload(payload)
    try:
        audio_bytes = encode_audio_bytes(waveform, output_format, sample_rate)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    loaded_devices = [device for _, device in MODEL_CACHE]
    return {
        "msg": "pong",
        "type": "VoxCPMTTS",
        "version": APP_VERSION,
        "build_id": BUILD_ID,
        "runtime": get_runtime_label(),
        "device": DEFAULT_DEVICE,
        "model_id": DEFAULT_MODEL_ID,
        "load_denoiser": DEFAULT_LOAD_DENOISER,
        "load_asr": DEFAULT_LOAD_ASR,
        "asr_model_id": ASR_MODEL_ID,
        "optimize": DEFAULT_OPTIMIZE,
        "languages": SUPPORTED_LANGUAGES,
        "loaded_model_devices": loaded_devices,
        "output_formats": get_supported_output_formats(),
        "stream_formats": STREAM_FORMATS,
    }


def create_ui() -> gr.Blocks:
    hardware_choices = get_hardware_choices()
    hardware_values = {value for _, value in hardware_choices}
    default_hardware = DEFAULT_DEVICE if DEFAULT_DEVICE in hardware_values else "auto"

    def generate_file(
        text: str,
        control: str,
        reference_audio: Optional[str],
        use_ref_text: bool,
        ref_text: str,
        cfg_value: float,
        inference_timesteps: int,
        normalize_text: bool,
        denoise: bool,
        output_format: str,
        hardware: str,
    ):
        payload = TTSRequest(
            text=text,
            control=None if use_ref_text else (control or None),
            ref_audio=reference_audio,
            ref_text=(ref_text or None) if use_ref_text else None,
            cfg_value=cfg_value,
            inference_timesteps=int(inference_timesteps),
            normalize=normalize_text,
            denoise=denoise,
            output_format=output_format,
            device=hardware,
        )
        output_format, sample_rate, waveform = synthesize_payload(payload)
        return encoded_audio_to_temp_file(waveform, output_format, sample_rate)

    def toggle_ultimate_mode(enabled: bool):
        if enabled:
            return (
                gr.update(visible=True),
                gr.update(interactive=False, value=""),
                gr.update(interactive=True),
            )
        return (
            gr.update(visible=False, value=""),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    def transcribe_for_ui(reference_audio: Optional[str]) -> str:
        if not reference_audio:
            raise gr.Error("Upload or record reference audio before transcription.")
        try:
            return transcribe_reference_audio(reference_audio)
        except Exception as exc:
            raise gr.Error(f"Reference transcription failed: {exc}") from exc

    with gr.Blocks(title="VoxCPMTTS") as ui:
        gr.HTML(
            "<div style='position:fixed;right:12px;top:12px;z-index:9999;"
            "background:rgba(0,0,0,.55);color:#fff;border-radius:8px;padding:6px 10px;"
            "font-size:12px'>"
            f"Version: {APP_VERSION} | Build: {BUILD_ID}<br>{get_runtime_label()}"
            "</div>"
        )
        gr.Markdown(
            "# VoxCPMTTS\n"
            "VoxCPM2 text to speech with voice design, controllable cloning, "
            "and transcript-guided cloning."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Target Text",
                    value="VoxCPM2 generates realistic multilingual speech with voice design and cloning.",
                    lines=4,
                )
                control = gr.Textbox(
                    label="Control Instruction",
                    placeholder="A calm warm narrator / young female, gentle and bright / faster and cheerful",
                    lines=2,
                )
                reference_audio = gr.Audio(
                    label="Reference Audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                use_ref_text = gr.Checkbox(
                    value=False,
                    label="Ultimate Cloning Mode",
                    info="Use a transcript with the reference audio. Control Instruction is disabled in this mode.",
                )
                ref_text = gr.Textbox(
                    label="Reference Transcript",
                    placeholder="Use the transcribe button or enter the reference transcript manually.",
                    lines=2,
                    visible=False,
                )
                transcribe_btn = gr.Button("Transcribe Reference", interactive=False)
                with gr.Row():
                    hardware = gr.Dropdown(hardware_choices, value=default_hardware, label="Hardware")
                    output_format = gr.Dropdown(
                        choices=[(config["label"], key) for key, config in OUTPUT_FORMATS.items()],
                        value="mp3",
                        label="Output Format",
                    )
                with gr.Accordion("Advanced", open=False):
                    cfg_value = gr.Slider(0.1, 10.0, value=2.0, step=0.1, label="CFG")
                    inference_timesteps = gr.Slider(1, 100, value=10, step=1, label="Inference Timesteps")
                    normalize_text = gr.Checkbox(value=False, label="Normalize Text")
                    denoise = gr.Checkbox(value=False, label="Denoise Reference Audio")
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Audio", interactive=False, autoplay=True)
                gr.Markdown(
                    "API docs: `/tts/docs`\n\n"
                    "Use `ref_audio` alone for controllable cloning, or enable Ultimate "
                    "Cloning Mode to pair reference audio with `ref_text`. VoxCPM2 "
                    "auto-detects its supported languages from the input text."
                )

        use_ref_text.change(
            fn=toggle_ultimate_mode,
            inputs=[use_ref_text],
            outputs=[ref_text, control, transcribe_btn],
        )
        transcribe_btn.click(
            fn=transcribe_for_ui,
            inputs=[reference_audio],
            outputs=[ref_text],
            show_progress=True,
        )
        generate_btn.click(
            fn=generate_file,
            inputs=[
                text,
                control,
                reference_audio,
                use_ref_text,
                ref_text,
                cfg_value,
                inference_timesteps,
                normalize_text,
                denoise,
                output_format,
                hardware,
            ],
            outputs=[audio_output],
            show_progress=True,
        )

    return ui


api = FastAPI(
    title="VoxCPMTTS Service API",
    description="HTTP API for Hangry Labs VoxCPMTTS.",
    version=APP_VERSION,
    openapi_url="/tts/openapi.json",
    docs_url="/tts/docs",
    redoc_url="/tts/redoc",
)


@api.get("/tts/ping")
def ping() -> dict:
    return {"msg": "pong", "type": "VoxCPMTTS", "version": APP_VERSION, "build_id": BUILD_ID}


@api.get("/tts/status")
def status() -> dict:
    return get_status_payload()


@api.get("/tts/defaults")
def defaults() -> dict:
    return {
        "text": "Hello from Hangry Labs VoxCPMTTS.",
        "language": "English",
        "voice": "auto",
        "device": "auto",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "output_formats": {"default": "wav", "available": get_supported_output_formats()},
        "stream_formats": {"default": "wav", "available": STREAM_FORMATS},
    }


@api.get("/tts/formats")
def formats() -> dict:
    return {"default": "wav", "formats": get_supported_output_formats(), "aliases": FORMAT_ALIASES}


@api.get("/tts/stream-formats")
def stream_formats() -> dict:
    return {
        "default": "wav",
        "formats": STREAM_FORMATS,
        "aliases": STREAM_FORMAT_ALIASES,
        "granularity": "full_synthesis_result",
    }


@api.get("/tts/languages")
def languages() -> dict:
    return {"languages": SUPPORTED_LANGUAGES, "dialect_note": "Chinese dialect generation is text/control driven."}


@api.get("/tts/voices")
def voices() -> dict:
    return {
        "voices": [
            {"id": "auto", "name": "Auto Voice Design", "language": "auto"},
            {"id": "reference", "name": "Reference Audio Clone", "language": "auto"},
        ],
        "note": "VoxCPM2 does not use a fixed speaker inventory. Use control/instruct or ref_audio.",
    }


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


@api.post("/tts/generate")
def generate_tts(payload: TTSRequest = Body(...)) -> StreamingResponse:
    return stream_audio_response(payload, "/tts/generate")


@api.post("/tts/convert")
def convert(payload: TTSRequest = Body(...)) -> StreamingResponse:
    return stream_audio_response(payload, "/tts/convert")


@api.post("/tts/stream")
def stream_tts(payload: StreamingTTSRequest = Body(...)) -> StreamingResponse:
    try:
        payload.output_format = normalize_stream_format(payload.stream_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return stream_audio_response(payload, "/tts/stream")


@api.post("/tts/purge")
def purge_models(payload: PurgeRequest | None = Body(None)) -> dict:
    requested_device = payload.device if payload else None
    if requested_device:
        device = resolve_requested_device(requested_device)
        purged = []
        for cache_key in list(MODEL_CACHE):
            if cache_key[1] == device:
                del MODEL_CACHE[cache_key]
                purged.append(device)
        return {"purged": purged, "remaining_model_devices": [device for _, device in MODEL_CACHE]}

    purged = [device for _, device in MODEL_CACHE]
    MODEL_CACHE.clear()
    return {"purged": purged, "remaining_model_devices": []}


app = gr.mount_gradio_app(api, create_ui(), path="/")


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8808"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run("voxcpm.app:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
