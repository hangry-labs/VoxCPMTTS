from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException, Response
from fastapi.testclient import TestClient

import voxcpm.app as runtime
from voxcpm.openai_compat import OpenAISpeechRequest


def test_openai_model_discovery_and_alias_lookup() -> None:
    with TestClient(runtime.app) as client:
        models = client.get("/v1/models")
        alias = client.get("/v1/models/tts-1")
        repository_alias = client.get("/v1/models/openbmb/VoxCPM2")

    assert models.status_code == 200
    assert models.json() == {
        "object": "list",
        "data": [{"id": "voxcpm2", "object": "model", "created": 0, "owned_by": "hangry-labs"}],
    }
    assert alias.status_code == 200
    assert alias.json()["alias_for"] == "voxcpm2"
    assert repository_alias.status_code == 200
    assert repository_alias.json()["id"] == "voxcpm2"


def test_openai_optional_bearer_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXCPMTTS_API_KEY", "local-secret")
    with TestClient(runtime.app) as client:
        missing = client.get("/v1/models")
        incorrect = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        accepted = client.get("/v1/models", headers={"Authorization": "Bearer local-secret"})

    assert missing.status_code == 401
    assert missing.json()["error"]["type"] == "authentication_error"
    assert missing.headers["www-authenticate"] == "Bearer"
    assert incorrect.status_code == 401
    assert accepted.status_code == 200


def test_openai_validation_and_not_found_errors_use_openai_envelope() -> None:
    with TestClient(runtime.app) as client:
        invalid = client.post("/v1/audio/speech", json={"model": "tts-1", "voice": "alloy"})
        missing = client.get("/v1/not-a-route")

    assert invalid.status_code == 400
    assert invalid.json()["error"]["param"] == "input"
    assert invalid.json()["error"]["code"] == "invalid_parameter"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    ("payload_update", "param", "code"),
    [
        ({"model": "unknown"}, "model", "model_not_found"),
        ({"voice": "unknown"}, "voice", "unsupported_voice"),
        ({"response_format": "ogg"}, "response_format", "unsupported_format"),
        ({"stream_format": "sse"}, "stream_format", "unsupported_parameter"),
    ],
)
def test_openai_rejects_unsupported_standard_options(
    payload_update: dict[str, str],
    param: str,
    code: str,
) -> None:
    payload = {"model": "tts-1", "input": "Hello.", "voice": "alloy"}
    payload.update(payload_update)
    with TestClient(runtime.app) as client:
        response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert response.json()["error"]["code"] == code


def test_openai_standard_request_maps_to_stable_voxcpm_request() -> None:
    captured: list[runtime.TTSRequest] = []

    def fake_audio_response(payload: runtime.TTSRequest, route_name: str) -> Response:
        captured.append(payload)
        assert route_name == "/v1/audio/speech"
        return Response(b"audio", media_type="audio/mpeg")

    with patch.object(runtime, "stream_audio_response", side_effect=fake_audio_response):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "model": "gpt-4o-mini-tts",
                    "input": "Hello from the compatibility API.",
                    "voice": "alloy",
                    "instructions": "Speak quietly.",
                    "response_format": "mp3",
                    "speed": 1.25,
                    "unknown_future_field": True,
                },
            )

    assert response.status_code == 200
    assert response.content == b"audio"
    request = captured[0]
    assert request.text == "Hello from the compatibility API."
    assert request.control == "Speak quietly."
    assert request.output_format == "mp3"
    assert request.speed == 1.25
    assert request.seed == 104_729
    assert request.randomize_seed is False
    assert request.normalize_loudness is True
    assert request.protect_long_audio is True


def test_openai_runtime_failure_uses_server_error_envelope() -> None:
    with patch.object(
        runtime,
        "stream_audio_response",
        side_effect=HTTPException(status_code=500, detail="Generation failed."),
    ):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"model": "tts-1", "input": "Hello.", "voice": "alloy"},
            )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "internal_server_error"


def test_openai_default_voice_is_stable_alloy_preset() -> None:
    request = runtime.openai_speech_to_tts_request(
        OpenAISpeechRequest(model="tts-1", input="Hello.", voice="default")
    )

    assert request.voice == "alloy"
    assert request.seed == 104_729
    assert request.randomize_seed is False


def test_openai_saved_voice_restores_profile_recipe(tmp_path: Path) -> None:
    (tmp_path / "profiles.json").write_text(
        json.dumps(
            {
                "joken": {
                    "profile_type": "designed",
                    "audio_file": "",
                    "design_audio_file": "",
                    "portrait_file": "",
                    "ref_text": "",
                    "control": "A precise Polish narrator.",
                    "description": "Test profile",
                    "tags": ["polish"],
                    "language": "Polish",
                    "recipe": {
                        "seed": 4242,
                        "randomize_seed": False,
                        "cfg_value": 2.5,
                        "inference_timesteps": 12,
                        "clone_mode": "reference",
                        "normalize": True,
                        "normalize_loudness": False,
                    },
                    "created_at": "2026-10-10T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )

    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        request = runtime.openai_speech_to_tts_request(
            OpenAISpeechRequest(model="voxcpm2", input="Dzien dobry.", voice={"id": "Joken"})
        )
        with TestClient(runtime.app) as client:
            voices = client.get("/v1/audio/voices")

    assert request.voice_profile == "joken"
    assert request.language == "Polish"
    assert request.seed == 4242
    assert request.randomize_seed is False
    assert request.cfg_value == 2.5
    assert request.inference_timesteps == 12
    assert request.clone_mode == "reference"
    assert request.normalize_text is True
    assert request.normalize_loudness is False
    saved = next(item for item in voices.json()["data"] if item["id"] == "joken")
    assert saved["profile_type"] == "designed"
    assert saved["owned_by"] == "local"
    assert "recipe" not in saved
    assert "ref_text" not in saved


def test_openai_extensions_override_saved_defaults() -> None:
    request = runtime.openai_speech_to_tts_request(
        OpenAISpeechRequest(
            model="voxcpmtts",
            input="Hello.",
            voice="nova",
            seed=99,
            randomize_seed=False,
            normalize_text=True,
            normalize_loudness=False,
            protect_long_audio=False,
            cfg_value=3.0,
            inference_timesteps=8,
            denoise=True,
            device="cuda:0",
            clone_mode="reference",
            ref_audio="/app/persistent/example.wav",
        )
    )

    assert request.seed == 99
    assert request.randomize_seed is False
    assert request.normalize_text is True
    assert request.normalize_loudness is False
    assert request.protect_long_audio is False
    assert request.cfg_value == 3.0
    assert request.inference_timesteps == 8
    assert request.denoise is True
    assert request.device == "cuda:0"
    assert request.ref_audio == "/app/persistent/example.wav"
    assert request.clone_mode == "reference"


@pytest.mark.parametrize(
    ("output_format", "magic"),
    [
        ("wav", b"RIFF"),
        ("mp3", b"ID3"),
        ("flac", b"fLaC"),
        ("opus", b"OggS"),
    ],
)
def test_openai_audio_formats_encode(output_format: str, magic: bytes) -> None:
    sample_rate = 48_000
    timeline = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 440 * timeline)

    encoded = runtime.encode_audio_bytes(audio, output_format, sample_rate)

    assert encoded.startswith(magic)


def test_openai_aac_and_pcm_formats_encode() -> None:
    sample_rate = 48_000
    timeline = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 440 * timeline)

    aac = runtime.encode_audio_bytes(audio, "aac", sample_rate)
    pcm = runtime.encode_audio_bytes(audio, "pcm", sample_rate)

    assert aac[0] == 0xFF and aac[1] & 0xF0 == 0xF0
    assert 4_600 <= len(pcm) <= 5_000


def test_raw_pcm_is_api_only_in_native_format_discovery() -> None:
    formats = runtime.get_supported_output_formats()

    assert formats["pcm"]["ui"] is False
    assert formats["opus"]["ui"] is True
    assert formats["aac"]["ui"] is True


@pytest.mark.parametrize(("speed", "duration"), [(0.5, 2.0), (2.0, 0.5)])
def test_openai_speed_preserves_pitch_and_changes_duration(speed: float, duration: float) -> None:
    sample_rate = 48_000
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 440 * timeline)

    adjusted = runtime.adjust_audio_speed(audio, sample_rate, speed)

    assert len(adjusted) / sample_rate == pytest.approx(duration, rel=0.08)


def test_openapi_schema_exposes_compatibility_route() -> None:
    with TestClient(runtime.app) as client:
        schema = client.get("/tts/openapi.json").json()

    assert "/v1/audio/speech" in schema["paths"]
    request_schema = schema["paths"]["/v1/audio/speech"]["post"]["requestBody"]
    assert request_schema["required"] is True
