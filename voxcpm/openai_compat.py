"""OpenAI speech API schemas, discovery metadata, authentication, and errors."""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


OPENAI_MODEL_ID = "voxcpm2"
OPENAI_MODEL_ALIASES = {
    "voxcpm": OPENAI_MODEL_ID,
    "voxcpm2": OPENAI_MODEL_ID,
    "voxcpmtts": OPENAI_MODEL_ID,
    "openbmb/voxcpm2": OPENAI_MODEL_ID,
    "tts-1": OPENAI_MODEL_ID,
    "tts-1-hd": OPENAI_MODEL_ID,
    "gpt-4o-mini-tts": OPENAI_MODEL_ID,
    "gpt-4o-mini-tts-2025-12-15": OPENAI_MODEL_ID,
}
OPENAI_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
OPENAI_VOICE_PRESETS: dict[str, tuple[str, int]] = {
    "alloy": ("A balanced, clear, and composed adult voice with natural pacing.", 104_729),
    "ash": ("A confident adult male voice with a calm, conversational delivery.", 130_363),
    "ballad": ("A warm, expressive storyteller with gentle emotional range.", 155_921),
    "coral": ("A friendly adult female voice with a bright, welcoming tone.", 181_081),
    "echo": ("A smooth adult male voice with measured pacing and a reflective tone.", 206_369),
    "fable": ("An animated narrator with expressive rhythm and a lightly theatrical style.", 231_533),
    "onyx": ("A deep, authoritative adult male voice with deliberate pacing.", 256_799),
    "nova": ("A lively adult female voice with energetic, polished delivery.", 282_083),
    "sage": ("A thoughtful, mature voice with patient and reassuring delivery.", 307_261),
    "shimmer": ("A gentle adult female voice with a soft, optimistic character.", 332_447),
    "verse": ("A versatile presenter with expressive but controlled delivery.", 357_643),
    "marin": ("A natural adult female voice with articulate, nuanced delivery.", 382_847),
    "cedar": ("A natural adult male voice with grounded, articulate delivery.", 408_041),
}


class OpenAISpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., min_length=1, description="OpenAI-compatible model id.")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize.")
    voice: str | dict[str, str] = Field(
        ...,
        description="OpenAI voice alias, saved VoxCPMTTS voice name, or custom voice id object.",
    )
    instructions: str | None = Field(
        None,
        max_length=4096,
        description="Optional delivery instructions mapped to VoxCPM voice direction.",
    )
    response_format: str = Field(
        "mp3",
        description="OpenAI audio format: mp3, opus, aac, flac, wav, or pcm.",
    )
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed multiplier.")
    stream_format: str = Field(
        "audio",
        description="Binary audio streaming. SSE speech events are not currently supported.",
    )

    # VoxCPMTTS extensions. Standard OpenAI clients can omit all of these.
    language: str = Field("English", description="VoxCPMTTS extension: language hint.")
    voice_profile: str | None = Field(None, description="VoxCPMTTS extension: saved voice profile.")
    seed: int | None = Field(None, ge=0, le=2**32 - 1, description="VoxCPMTTS extension: fixed seed.")
    randomize_seed: bool | None = Field(None, description="VoxCPMTTS extension: randomize the seed.")
    normalize_text: bool = Field(False, description="VoxCPMTTS extension: normalize plain text.")
    normalize_loudness: bool = Field(True, description="VoxCPMTTS extension: normalize toward -16 LUFS.")
    protect_long_audio: bool = Field(True, description="VoxCPMTTS extension: protect long-form synthesis.")
    cfg_value: float = Field(2.0, ge=0.1, le=10.0, description="VoxCPMTTS extension: guidance scale.")
    inference_timesteps: int | None = Field(None, ge=1, le=100, description="VoxCPMTTS extension: inference steps.")
    device: str = Field("auto", description="VoxCPMTTS extension: inference device.")
    denoise: bool = Field(False, description="VoxCPMTTS extension: denoise reference audio.")
    ref_audio: str | None = Field(None, description="VoxCPMTTS extension: container-visible reference audio.")
    ref_text: str | None = Field(None, description="VoxCPMTTS extension: reference transcript.")
    clone_mode: str = Field("auto", description="VoxCPMTTS extension: auto, reference, or transcript.")


class OpenAIAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        self.headers = headers or {}


def openai_error_response(
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=headers,
    )


def require_openai_api_key(request: Request) -> None:
    configured_key = os.getenv("VOXCPMTTS_API_KEY", "").strip()
    if not configured_key:
        return
    authorization = request.headers.get("authorization", "")
    scheme, _, provided_key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(provided_key, configured_key):
        raise OpenAIAPIError(
            "Incorrect API key provided.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def resolve_openai_model(model: str) -> str:
    normalized = model.strip().lower()
    resolved = OPENAI_MODEL_ALIASES.get(normalized)
    if resolved is None:
        raise OpenAIAPIError(
            f"The model '{model}' does not exist.",
            param="model",
            code="model_not_found",
        )
    return resolved


def resolve_openai_voice_id(voice: str | dict[str, str]) -> str:
    if isinstance(voice, dict):
        voice_id = str(voice.get("id", "")).strip()
        if not voice_id:
            raise OpenAIAPIError(
                "Custom voice objects require a non-empty 'id'.",
                param="voice",
                code="unsupported_voice",
            )
        return voice_id
    voice_id = voice.strip()
    if not voice_id:
        raise OpenAIAPIError(
            "Voice must not be empty.",
            param="voice",
            code="unsupported_voice",
        )
    return voice_id


def normalize_openai_response_format(response_format: str) -> str:
    normalized = response_format.strip().lower()
    if normalized not in OPENAI_RESPONSE_FORMATS:
        supported = ", ".join(sorted(OPENAI_RESPONSE_FORMATS))
        raise OpenAIAPIError(
            f"Unsupported response_format '{response_format}'. Supported formats: {supported}.",
            param="response_format",
            code="unsupported_format",
        )
    return normalized


def validate_openai_stream_format(stream_format: str) -> None:
    if stream_format.strip().lower() != "audio":
        raise OpenAIAPIError(
            "Only stream_format='audio' is supported. SSE speech events are not implemented.",
            param="stream_format",
            code="unsupported_parameter",
        )


def openai_model_payload(requested_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": OPENAI_MODEL_ID,
        "object": "model",
        "created": 0,
        "owned_by": "hangry-labs",
    }
    if requested_id and requested_id.strip().lower() != OPENAI_MODEL_ID:
        payload["requested_id"] = requested_id
        payload["alias_for"] = OPENAI_MODEL_ID
    return payload


def openai_voice_payload(voice_id: str, *, profile_type: str, owned_by: str) -> dict[str, Any]:
    return {
        "id": voice_id,
        "object": "voice",
        "owned_by": owned_by,
        "profile_type": profile_type,
    }
