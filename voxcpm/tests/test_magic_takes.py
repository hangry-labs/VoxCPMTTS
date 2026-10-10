from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import voxcpm.app as runtime
from voxcpm.magic_takes import MAGIC_TAKE_MEDIA_TYPE
from voxcpm.ssml_execution import assemble_audio_timeline


class _FakeTTSModel:
    sample_rate = 16_000


class _FakeModel:
    def __init__(self) -> None:
        self.tts_model = _FakeTTSModel()
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        amplitude = 0.05 + (int(kwargs["seed"]) % 10) / 100
        return np.full(16_000, amplitude, dtype=np.float32)


DOCUMENT = """<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
  xmlns:h="https://hangrylabs.app/ns/ssml-h/1.0" xml:lang="en-US">
  First turn.
  <break time="250ms" />
  Second turn.
</speak>"""


def _payload() -> dict:
    return {
        "text": DOCUMENT,
        "input_type": "ssml-h",
        "output_format": "wav",
        "randomize_seed": False,
        "seed": 42,
    }


def _unpack_bundle(data: bytes) -> tuple[dict, bytes]:
    manifest_size = struct.unpack_from("<I", data)[0]
    manifest_start = 4
    manifest_end = manifest_start + manifest_size
    manifest = json.loads(data[manifest_start:manifest_end])
    return manifest, data[manifest_end:]


def _bundle_audio(manifest: dict, payload: bytes, item: dict) -> bytes:
    start = item["offset"]
    return payload[start:start + item["length"]]


def _wav_frames(data: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(data), "rb") as source:
        return source.getframerate(), source.getnframes()


def test_canonical_timeline_assembly_honors_explicit_break() -> None:
    first = np.full(100, 1_000, dtype=np.int16)
    second = np.full(100, 2_000, dtype=np.int16)

    waveform = assemble_audio_timeline(
        iter((("speech", first), ("break", 250), ("speech", second))),
        1_000,
    )

    assert waveform.size == 450
    assert np.count_nonzero(waveform[100:350]) == 0


def test_magic_render_returns_individual_takes_and_composed_output(tmp_path: Path) -> None:
    model = _FakeModel()
    with (
        patch.object(runtime, "SSML_STAGING_DIR", tmp_path / "staging"),
        patch.object(runtime, "get_model", return_value=model),
        TestClient(runtime.app) as client,
    ):
        response = client.post(
            "/tts/magic/render",
            data={
                "payload": json.dumps(_payload()),
                "speech_ids": json.dumps(["turn-1", "turn-2"]),
                "retained": "[]",
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(MAGIC_TAKE_MEDIA_TYPE)
    manifest, binary = _unpack_bundle(response.content)
    assert [item["id"] for item in manifest["takes"]] == ["turn-1", "turn-2"]
    assert [item["seed"] for item in manifest["takes"]] == [42, 43]
    assert [call["text"] for call in model.calls] == ["First turn.", "Second turn."]
    assert _wav_frames(_bundle_audio(manifest, binary, manifest["takes"][0])) == (16_000, 16_000)
    assert _wav_frames(_bundle_audio(manifest, binary, manifest["output"]))[0] == 16_000


def test_magic_render_reuses_locked_take_and_generates_only_unlocked_turn(tmp_path: Path) -> None:
    initial_model = _FakeModel()
    with (
        patch.object(runtime, "SSML_STAGING_DIR", tmp_path / "staging"),
        patch.object(runtime, "get_model", return_value=initial_model),
        TestClient(runtime.app) as client,
    ):
        initial = client.post(
            "/tts/magic/render",
            data={"payload": json.dumps(_payload()), "speech_ids": '["turn-1","turn-2"]'},
        )
    initial_manifest, initial_binary = _unpack_bundle(initial.content)
    locked = _bundle_audio(initial_manifest, initial_binary, initial_manifest["takes"][0])

    next_model = _FakeModel()
    with (
        patch.object(runtime, "SSML_STAGING_DIR", tmp_path / "staging"),
        patch.object(runtime, "get_model", return_value=next_model),
        TestClient(runtime.app) as client,
    ):
        response = client.post(
            "/tts/magic/render",
            data={
                "payload": json.dumps(_payload()),
                "speech_ids": '["turn-1","turn-2"]',
                "retained": '[{"id":"turn-1","seed":42}]',
            },
            files=[("takes", ("turn-1.wav", locked, "audio/wav"))],
        )

    assert response.status_code == 200, response.text
    manifest, binary = _unpack_bundle(response.content)
    assert len(next_model.calls) == 1
    assert next_model.calls[0]["text"] == "Second turn."
    assert _bundle_audio(manifest, binary, manifest["takes"][0]) == locked


def test_magic_render_with_all_retained_takes_does_not_load_model(tmp_path: Path) -> None:
    model = _FakeModel()
    with (
        patch.object(runtime, "SSML_STAGING_DIR", tmp_path / "staging"),
        patch.object(runtime, "get_model", return_value=model),
        TestClient(runtime.app) as client,
    ):
        initial = client.post(
            "/tts/magic/render",
            data={"payload": json.dumps(_payload()), "speech_ids": '["turn-1","turn-2"]'},
        )
    manifest, binary = _unpack_bundle(initial.content)
    audio = [_bundle_audio(manifest, binary, item) for item in manifest["takes"]]

    with patch.object(runtime, "get_model") as get_model, TestClient(runtime.app) as client:
        response = client.post(
            "/tts/magic/render",
            data={
                "payload": json.dumps(_payload()),
                "speech_ids": '["turn-1","turn-2"]',
                "retained": '[{"id":"turn-1","seed":42},{"id":"turn-2","seed":43}]',
            },
            files=[
                ("takes", ("turn-1.wav", audio[0], "audio/wav")),
                ("takes", ("turn-2.wav", audio[1], "audio/wav")),
            ],
        )

    assert response.status_code == 200, response.text
    get_model.assert_not_called()


def test_magic_single_take_and_model_free_reassembly(tmp_path: Path) -> None:
    model = _FakeModel()
    with (
        patch.object(runtime, "SSML_STAGING_DIR", tmp_path / "staging"),
        patch.object(runtime, "get_model", return_value=model),
        TestClient(runtime.app) as client,
    ):
        first = client.post("/tts/magic/take", json={"payload": _payload(), "speech_index": 0})
        second = client.post("/tts/magic/take", json={"payload": _payload(), "speech_index": 1})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.headers["x-voxcpm-seed"] == "42"
    assert second.headers["x-voxcpm-seed"] == "43"
    assert [call["text"] for call in model.calls] == ["First turn.", "Second turn."]

    options = {
        "items": [
            {"type": "speech", "id": "turn-1"},
            {"type": "break", "milliseconds": 500},
            {"type": "speech", "id": "turn-2"},
        ],
        "output_format": "wav",
    }
    with patch.object(runtime, "get_model") as get_model, TestClient(runtime.app) as client:
        assembled = client.post(
            "/tts/magic/assemble-upload",
            data={
                "options": json.dumps(options),
                "take_metadata": '[{"id":"turn-1","seed":42},{"id":"turn-2","seed":43}]',
            },
            files=[
                ("takes", ("turn-1.wav", first.content, "audio/wav")),
                ("takes", ("turn-2.wav", second.content, "audio/wav")),
            ],
        )

    assert assembled.status_code == 200, assembled.text
    assert _wav_frames(assembled.content) == (16_000, 40_000)
    get_model.assert_not_called()
