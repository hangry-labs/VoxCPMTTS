from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

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
                    "/assets/voxcpmtts_mascot.webp",
                    "/assets/hangrylabs_logo.webp",
                    "/system/gpu",
                )
            }

    assert all(response.status_code == 200 for response in responses.values())
    assert "VoxCPMTTS" in responses["/"].text
    assert "voxcpmtts_logo_horizontal.webp" in responses["/"].text
    assert 'src="/assets/voxcpmtts_mascot.webp"' in responses["/"].text
    assert "UI v" not in responses["/"].text
    assert 'data-tab="clone"' in responses["/"].text
    assert 'data-tab="voices"' in responses["/"].text
    assert 'id="voice-mode"' not in responses["/"].text
    assert 'id="voice-design-details"' in responses["/"].text
    assert 'id="reference-text"' in responses["/"].text
    assert 'id="voice-profile"' in responses["/"].text
    assert 'id="generate-progress"' in responses["/"].text
    assert 'id="randomize-seed"' in responses["/"].text
    assert 'id="generation-settings"' in responses["/"].text
    assert 'id="generate-timestamps"' in responses["/"].text
    assert 'id="stream-live-wave"' in responses["/"].text
    assert 'id="reference-record-wave"' in responses["/"].text
    assert 'id="profile-audio-drop"' in responses["/"].text
    assert 'data-input-type="ssml"' in responses["/"].text
    assert 'data-input-type="ssml-h"' in responses["/"].text
    assert 'id="profile-edit-cancel"' in responses["/"].text
    assert "gradio" not in responses["/"].text.lower()
    assert responses["/system/gpu"].json()["gpus"] == []
    assert responses["/system/gpu"].headers["cache-control"] == "no-store"

    script = responses["/static/app.js"].text
    assert "IncrementalAudioPlayback" in script
    assert "response.body.getReader()" in script
    assert "await playback.append(value)" in script
    assert "class StreamWaveform" in script
    assert "fetchJson('/tts/activity'" in script
    assert "GPU_HISTORY_RETENTION_MS = 10 * 60 * 1000" in script
    assert "function renderGpuMonitor(" in script
    assert "function stopGpuMonitor(" in script
    assert "inputDrafts: { text: null, ssml: null, 'ssml-h': null }" in script
    assert "input_type: state.inputType" in script
    assert "openProfileEditor(profile)" in script
    assert "sessionStorage.setItem(GPU_SESSION_KEY" in script


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

    def fake_chunks(payload):
        path = Path(payload.ref_audio)
        observed.update(
            path=path,
            exists=path.exists(),
            contents=path.read_bytes(),
            stream_format=payload.stream_format,
        )
        return "mp3", 24_000, iter([b"model-chunk"]), 123

    def fake_encoder(chunks, output_format, sample_rate):
        observed.update(output_format=output_format, sample_rate=sample_rate, chunks=list(chunks))
        yield b"first-"
        yield b"second"

    request = {"text": "Stream this.", "stream_format": "mp3"}
    with (
        patch.object(runtime, "synthesize_payload_chunks", side_effect=fake_chunks),
        patch.object(runtime, "encode_audio_stream", side_effect=fake_encoder),
    ):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/stream-upload",
                data={"payload": json.dumps(request)},
                files={"reference_audio": ("reference.wav", b"reference-bytes", "audio/wav")},
            )

    assert response.status_code == 200
    assert response.content == b"first-second"
    assert response.headers["x-voxcpm-streaming"] == "progressive-chunks"
    assert response.headers["x-voxcpm-format"] == "mp3"
    assert response.headers["x-voxcpm-seed"] == "123"
    assert observed["exists"] is True
    assert observed["contents"] == b"reference-bytes"
    assert observed["stream_format"] == "mp3"
    assert observed["output_format"] == "mp3"
    assert observed["sample_rate"] == 24_000
    assert observed["chunks"] == [b"model-chunk"]
    assert not observed["path"].exists()


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


def test_reference_asr_loads_pinned_whisper_lazily_and_reuses_it(tmp_path: Path) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"audio")
    asr_pipeline = Mock(return_value={"text": "  Reference transcript  "})

    with (
        patch.object(runtime, "ASR_MODEL", None),
        patch.object(runtime, "DEFAULT_LOAD_ASR", True),
        patch.object(runtime, "DEFAULT_LOCAL_ONLY", True),
        patch.object(runtime, "ASR_DEVICE", "cpu"),
        patch("huggingface_hub.snapshot_download", return_value="/models/whisper-base") as download,
        patch.object(runtime, "_create_asr_pipeline", return_value=asr_pipeline) as create_pipeline,
    ):
        first = runtime.transcribe_reference_audio(str(audio_path), "auto")
        second = runtime.transcribe_reference_audio(str(audio_path), "English")

    assert first == "Reference transcript"
    assert second == "Reference transcript"
    download.assert_called_once_with(
        repo_id=runtime.ASR_MODEL_ID,
        revision=runtime.ASR_MODEL_REVISION,
        allow_patterns=list(runtime.ASR_ALLOW_PATTERNS),
        local_files_only=True,
    )
    create_pipeline.assert_called_once_with("/models/whisper-base", "cpu")
    assert asr_pipeline.call_args_list[0].kwargs["generate_kwargs"] == {"task": "transcribe"}
    assert asr_pipeline.call_args_list[1].kwargs["generate_kwargs"] == {
        "task": "transcribe",
        "language": "english",
    }


def test_upload_rejects_unsupported_file_type() -> None:
    with TestClient(runtime.app) as client:
        response = client.post(
            "/tts/generate-upload",
            data={"payload": json.dumps({"text": "Hello"})},
            files={"reference_audio": ("reference.txt", b"not-audio", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported reference audio file type"


def test_saved_clone_profile_can_be_listed_resolved_and_deleted(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            created = client.post(
                "/tts/voice-profiles",
                data={
                    "name": "Studio Narrator",
                    "profile_type": "cloned",
                    "description": "Warm studio reference",
                    "ref_text": "This is the original recording.",
                    "language": "English",
                },
                files={"reference_audio": ("reference.wav", b"reference-bytes", "audio/wav")},
            )
            listed = client.get("/tts/voice-profiles")
            audio = client.get("/tts/voice-profiles/studio-narrator/audio")

            payload = runtime.TTSRequest(text="A new sentence.", voice_profile="studio-narrator")
            kwargs = runtime.build_generate_kwargs(payload, object())

            deleted = client.delete("/tts/voice-profiles/studio-narrator")
            missing = client.get("/tts/voice-profiles/studio-narrator/audio")

    assert created.status_code == 200
    assert created.json()["id"] == "studio-narrator"
    assert created.json()["profile_type"] == "cloned"
    assert listed.json()["count"] == 1
    assert audio.content == b"reference-bytes"
    assert Path(kwargs["reference_wav_path"]).parent == tmp_path
    assert kwargs["prompt_wav_path"] == kwargs["reference_wav_path"]
    assert kwargs["prompt_text"] == "This is the original recording."
    assert 0 <= kwargs["seed"] <= runtime.MAX_RANDOM_SEED
    assert deleted.json() == {"deleted": "studio-narrator"}
    assert missing.status_code == 404


def test_saved_clone_profile_can_be_edited_without_replacing_audio(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            created = client.post(
                "/tts/voice-profiles",
                data={"name": "Studio Voice", "profile_type": "cloned", "ref_text": "Original."},
                files={"reference_audio": ("reference.wav", b"reference-bytes", "audio/wav")},
            )
            updated = client.put(
                "/tts/voice-profiles/studio-voice",
                data={
                    "description": "Updated studio voice",
                    "ref_text": "Updated transcript.",
                    "language": "English",
                },
            )
            audio = client.get("/tts/voice-profiles/studio-voice/audio")

    assert created.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["description"] == "Updated studio voice"
    assert updated.json()["ref_text"] == "Updated transcript."
    assert audio.content == b"reference-bytes"


def test_saved_voice_design_applies_its_control(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            created = client.post(
                "/tts/voice-profiles",
                data={
                    "name": "Calm Guide",
                    "profile_type": "designed",
                    "control": "A calm, warm guide with measured pacing",
                    "language": "English",
                },
            )
        payload = runtime.TTSRequest(text="Welcome.", voice_profile="calm-guide")
        kwargs = runtime.build_generate_kwargs(payload, object())

    assert created.status_code == 200
    assert created.json()["profile_type"] == "designed"
    assert kwargs["text"] == "(A calm, warm guide with measured pacing)Welcome."
    assert kwargs["reference_wav_path"] is None


def test_fixed_seed_is_forwarded_to_generation_kwargs() -> None:
    payload = runtime.TTSRequest(text="A repeatable sentence.", seed=1234, randomize_seed=False)

    kwargs = runtime.build_generate_kwargs(payload, object())

    assert kwargs["seed"] == 1234


def test_timestamp_upload_reports_unavailable_backend() -> None:
    with patch.object(runtime, "timestamp_backend_available", return_value=False):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/timestamps-upload",
                data={"text": "Hello", "level": "word"},
                files={"audio": ("generated.wav", b"RIFF", "audio/wav")},
            )

    assert response.status_code == 503
    assert "stable-ts" in response.json()["detail"]
