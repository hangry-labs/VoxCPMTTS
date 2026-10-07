from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

import voxcpm.app as runtime


def test_static_workspace_and_assets_are_available() -> None:
    with patch.object(runtime.GPU_MONITOR, "request_snapshot", return_value={"gpus": [], "history": {}}):
        with TestClient(runtime.app) as client:
            responses = {
                path: client.get(path)
                for path in (
                    "/",
                    "/static/app.js",
                    "/static/styles.css",
                    "/static/audio-editor.js",
                    "/static/vendor/lucide/lucide.woff2",
                    "/assets/voxcpmtts_logo_horizontal.webp",
                    "/assets/hangrylabs_logo.webp",
                    "/system/gpu",
                )
            }

    assert all(response.status_code == 200 for response in responses.values())
    assert "VoxCPMTTS" in responses["/"].text
    assert "voxcpmtts_logo_horizontal.webp" in responses["/"].text
    assert "gradio" not in responses["/"].text.lower()
    assert responses["/system/gpu"].json()["gpus"] == []


def test_generate_upload_passes_a_temporary_reference_and_removes_it() -> None:
    observed: dict[str, object] = {}

    def fake_response(payload, route_name):
        path = Path(payload.ref_audio)
        observed.update(
            path=path,
            exists=path.exists(),
            contents=path.read_bytes(),
            control=payload.control,
            ref_text=payload.ref_text,
            route=route_name,
        )
        return StreamingResponse(
            io.BytesIO(b"audio"),
            media_type="audio/mpeg",
            headers={"X-VoxCPM-Format": "mp3"},
        )

    request = {
        "text": "Hello from the browser.",
        "control": "warm narrator",
        "ref_text": None,
        "output_format": "mp3",
    }
    with patch.object(runtime, "stream_audio_response", side_effect=fake_response):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/generate-upload",
                data={"payload": json.dumps(request)},
                files={"reference_audio": ("reference.wav", b"reference-bytes", "audio/wav")},
            )

    assert response.status_code == 200
    assert response.content == b"audio"
    assert response.headers["x-voxcpm-format"] == "mp3"
    assert observed["exists"] is True
    assert observed["contents"] == b"reference-bytes"
    assert observed["control"] == "warm narrator"
    assert observed["route"] == "/tts/generate-upload"
    assert not observed["path"].exists()


def test_stream_upload_applies_stream_format() -> None:
    observed: dict[str, object] = {}

    def fake_response(payload, route_name):
        observed.update(output_format=payload.output_format, route=route_name)
        return StreamingResponse(io.BytesIO(b"wav"), media_type="audio/wav")

    request = {"text": "Stream this.", "stream_format": "wav"}
    with patch.object(runtime, "stream_audio_response", side_effect=fake_response):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/stream-upload",
                data={"payload": json.dumps(request)},
                files={"reference_audio": ("reference.wav", b"reference-bytes", "audio/wav")},
            )

    assert response.status_code == 200
    assert observed == {"output_format": "wav", "route": "/tts/stream-upload"}


def test_transcribe_upload_uses_and_removes_temporary_audio() -> None:
    observed: dict[str, object] = {}

    def fake_transcribe(path, language):
        audio_path = Path(path)
        observed.update(path=audio_path, exists=audio_path.exists(), language=language)
        return "Reference transcript"

    with patch.object(runtime, "transcribe_reference_audio", side_effect=fake_transcribe):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/transcribe-upload",
                data={"language": "auto"},
                files={"reference_audio": ("reference.mp3", b"reference-bytes", "audio/mpeg")},
            )

    assert response.status_code == 200
    assert response.json()["text"] == "Reference transcript"
    assert observed["exists"] is True
    assert observed["language"] == "auto"
    assert not observed["path"].exists()


def test_upload_rejects_unsupported_file_type() -> None:
    with TestClient(runtime.app) as client:
        response = client.post(
            "/tts/generate-upload",
            data={"payload": json.dumps({"text": "Hello"})},
            files={"reference_audio": ("reference.txt", b"not-audio", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported reference audio file type"
