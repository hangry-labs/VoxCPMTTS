from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import voxcpm.app as runtime
from voxcpm.ssml import SSML_H_NAMESPACE, SSMLValidationError, compile_ssml, ssml_capabilities


class _FakeTTSModel:
    sample_rate = 16_000


class _FakeModel:
    def __init__(self) -> None:
        self.tts_model = _FakeTTSModel()
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.full(16_000, 0.1, dtype=np.float32)


def test_shared_parser_preserves_order_and_explicit_zero_break() -> None:
    plan = compile_ssml(
        '<speak xml:lang="en-US">Hello.<break time="0ms"/><prosody rate="slow">World.</prosody></speak>',
        "ssml",
        resolve_language=runtime.resolve_ssml_language,
    )

    assert [unit.kind for unit in plan.units] == ["speech", "break", "speech"]
    assert plan.units[1].duration_ms == 0
    assert plan.units[2].prosody.rate == 0.75


def test_capabilities_match_voxcpm_support() -> None:
    capabilities = ssml_capabilities()

    assert capabilities["ssml_h"]["namespace"] == SSML_H_NAMESPACE
    assert capabilities["ssml_h"]["description_supported"] is True
    assert capabilities["ssml"]["phoneme_alphabets"] == []
    assert capabilities["processor"]["remote_audio"] is False


def test_unknown_saved_voice_is_rejected_before_model_loading() -> None:
    document = '<speak><voice name="missing">Hello.</voice></speak>'
    with patch.object(runtime, "get_model") as get_model:
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/generate",
                json={"text": document, "input_type": "ssml", "output_format": "wav"},
            )

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
    get_model.assert_not_called()


def test_standard_ssml_generates_ordered_units() -> None:
    model = _FakeModel()
    document = (
        '<speak xml:lang="en-US">Hello.<break time="0ms"/>'
        '<prosody rate="100%">World.</prosody></speak>'
    )
    with patch.object(runtime, "get_model", return_value=model):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/generate",
                json={
                    "text": document,
                    "input_type": "ssml",
                    "output_format": "wav",
                    "randomize_seed": False,
                    "seed": 42,
                },
            )

    assert response.status_code == 200, response.text
    assert response.headers["x-voxcpm-input-type"] == "ssml"
    assert [call["text"] for call in model.calls] == ["Hello.", "World."]
    assert [call["seed"] for call in model.calls] == [42, 43]


def test_ssml_h_profile_is_published_only_after_success() -> None:
    model = _FakeModel()
    document = f"""<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
      xmlns:h="{SSML_H_NAMESPACE}" xml:lang="en-US">
      <metadata><h:extensions version="1.0">
        <h:voice-definition name="Host" scope="profile" seed="7">
          <h:description>A warm and confident female host.</h:description>
          <h:sample xml:lang="en-US">Welcome to the show.</h:sample>
        </h:voice-definition>
      </h:extensions></metadata>
      <voice name="Host">Today we are testing SSML-H.</voice>
    </speak>"""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            patch.object(runtime, "VOICE_PROFILE_DIR", root / "voices"),
            patch.object(runtime, "SSML_STAGING_DIR", root / "staging"),
            patch.object(runtime, "get_model", return_value=model),
        ):
            with TestClient(runtime.app) as client:
                response = client.post(
                    "/tts/generate",
                    json={"text": document, "input_type": "ssml-h", "output_format": "wav"},
                )
                profiles = client.get("/tts/voice-profiles").json()["data"]
                duplicate = client.post(
                    "/tts/generate",
                    json={"text": document, "input_type": "ssml-h", "output_format": "wav"},
                )
        staging = list((root / "staging").glob("ssml-h-*"))

    assert response.status_code == 200, response.text
    assert profiles[0]["id"] == "host"
    assert profiles[0]["ref_text"] == "Welcome to the show."
    assert duplicate.status_code == 400
    assert "already exists" in duplicate.json()["detail"]
    assert len(model.calls) == 2
    assert staging == []


def test_invalid_ssml_mode_does_not_infer_markup() -> None:
    with patch.object(runtime, "get_model", return_value=_FakeModel()):
        payload = runtime.TTSRequest(text="<speak>Hello.</speak>")
    assert payload.input_type == "text"

    with np.testing.assert_raises(SSMLValidationError):
        compile_ssml(
            f'<speak xmlns:h="{SSML_H_NAMESPACE}"><metadata><h:extensions version="1.0"/></metadata>Hello.</speak>',
            "ssml",
        )
