from __future__ import annotations

import io
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

import voxcpm.app as runtime

TEST_PORTRAIT_WEBP = (Path(__file__).parents[2] / "assets" / "voxcpmtts_mascot.webp").read_bytes()


def encoded_reference_audio(tmp_path: Path, suffix: str) -> bytes:
    sample_rate = 48_000
    duration = 0.2
    samples = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    mono = 0.2 * np.sin(2 * np.pi * 440 * samples)
    stereo = np.column_stack((mono, mono * 0.5))
    source = tmp_path / f"source-{suffix.removeprefix('.')}"
    source = source.with_suffix(".wav")
    target = tmp_path / f"reference{suffix}"
    sf.write(source, stereo, sample_rate, subtype="PCM_16")

    codec_args = {
        ".m4a": ["-c:a", "aac", "-b:a", "128k"],
        ".aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
        ".webm": ["-c:a", "libopus", "-b:a", "128k"],
    }[suffix]
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            *codec_args,
            str(target),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return target.read_bytes()


def test_static_workspace_and_assets_are_available() -> None:
    with patch.object(runtime.GPU_MONITOR, "request_snapshot", return_value={"gpus": [], "history": {}}):
        with TestClient(runtime.app) as client:
            responses = {
                path: client.get(path)
                for path in (
                    "/",
                    "/en",
                    "/nb",
                    "/pl",
                    "/ja",
                    "/zh",
                    "/es",
                    "/static/app.js",
                    "/static/i18n.js",
                    "/static/magic-editor.js",
                    "/static/magic-document.js",
                    "/static/styles.css",
                    "/static/audio-editor.js",
                    "/static/locales/en.json",
                    "/static/locales/pl.json",
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
    assert 'data-i18n="tabs.design">Design</button>' in responses["/"].text
    assert 'data-tab="stream"' not in responses["/"].text
    assert 'data-panel="stream"' not in responses["/"].text
    assert 'data-generation-mode="generate"' in responses["/"].text
    assert 'data-generation-mode="stream"' in responses["/"].text
    assert 'id="stream-start"' not in responses["/"].text
    assert '<html lang="en" dir="ltr">' in responses["/"].text
    assert '<html lang="pl" dir="ltr">' in responses["/pl"].text
    assert '"locale":"pl"' in responses["/pl"].text
    assert '"storageKey":"voxcpmtts-ui-locale-v1"' in responses["/pl"].text
    assert '"tabs.generate":"Generuj"' in responses["/pl"].text
    assert 'data-tab="voices"' not in responses["/"].text
    assert 'id="voice-mode"' not in responses["/"].text
    assert 'id="voice-design-details"' not in responses["/"].text
    assert 'id="design-text-input"' in responses["/"].text
    assert 'id="design-text-input" type="text"' in responses["/"].text
    assert 'id="design-seed-lock"' in responses["/"].text
    assert 'id="design-voice-name"' in responses["/"].text
    assert 'id="design-voice-tags"' in responses["/"].text
    assert 'id="design-portrait-input"' in responses["/"].text
    assert 'class="portrait-placeholder"' in responses["/"].text
    assert 'id="clone-output-section"' in responses["/"].text
    assert 'id="voice-finishing"' in responses["/"].text
    assert '<option value="signalsmith"' in responses["/"].text
    assert 'id="signalsmith-controls"' in responses["/"].text
    assert 'id="post-pitch"' in responses["/"].text
    assert 'id="post-speed"' in responses["/"].text
    assert 'id="clone-processed-output"' in responses["/"].text
    assert 'data-save-version="original"' in responses["/"].text
    assert 'data-save-version="processed"' in responses["/"].text
    assert 'id="quick-profile-name"' not in responses["/"].text
    assert 'data-design-source="reference"' in responses["/"].text
    assert 'data-design-source="direction"' in responses["/"].text
    assert 'id="reference-text"' in responses["/"].text
    assert 'id="voice-profile"' in responses["/"].text
    assert 'id="generate-progress"' in responses["/"].text
    assert 'id="randomize-seed"' in responses["/"].text
    assert 'id="generation-settings"' in responses["/"].text
    assert 'id="normalize-loudness"' in responses["/"].text
    assert 'id="normalize-text"' in responses["/"].text
    assert 'id="timing-settings"' in responses["/"].text
    assert 'id="generate-timestamps"' in responses["/"].text
    assert 'id="stream-live-wave"' in responses["/"].text
    assert responses["/"].text.count('id="generate-progress"') == 1
    assert 'id="reference-record-wave"' in responses["/"].text
    assert 'class="reference-audio-surface"' in responses["/"].text
    assert 'id="reference-audio-choose"' in responses["/"].text
    assert 'id="reference-audio-clear"' in responses["/"].text
    assert responses["/"].text.count('id="transcribe-reference"') == 1
    assert 'data-clone-mode="reference"' in responses["/"].text
    assert 'data-clone-mode="transcript"' in responses["/"].text
    assert 'id="clone-profile-list"' in responses["/"].text
    assert 'id="clone-profile-filter"' in responses["/"].text
    assert 'id="store-generated-voice"' in responses["/"].text
    assert 'id="cancel-voice-edit"' in responses["/"].text
    assert 'id="update-profile-dialog"' in responses["/"].text
    assert 'id="profile-audio-drop"' not in responses["/"].text
    assert 'data-input-type="ssml"' in responses["/"].text
    assert 'data-input-type="ssml-h"' in responses["/"].text
    assert 'data-input-type="magic"' in responses["/"].text
    assert 'id="magic-editor-shell"' in responses["/"].text
    assert 'id="magic-expression-control"' in responses["/"].text
    assert 'id="magic-character-dialog"' in responses["/"].text
    assert 'id="profile-edit-dialog"' not in responses["/"].text
    assert "gradio" not in responses["/"].text.lower()
    assert responses["/system/gpu"].json()["gpus"] == []
    assert responses["/system/gpu"].headers["cache-control"] == "no-store"

    script = responses["/static/app.js"].text
    assert "IncrementalAudioPlayback" in script
    assert "response.body.getReader()" in script
    assert "await playback.append(value)" in script
    assert "startActivityPolling('stream')" in script
    assert "workflow === 'stream' ? 'generate' : workflow" in script
    assert "class StreamWaveform" in script
    assert "fetchJson('/tts/activity'" in script
    assert "GPU_HISTORY_RETENTION_MS = 10 * 60 * 1000" in script
    assert "function renderGpuMonitor(" in script
    assert "function stopGpuMonitor(" in script
    assert "inputDrafts: { text: null, ssml: null, 'ssml-h': null }" in script
    assert "state.inputType === 'magic' ? 'ssml-h' : state.inputType" in script
    assert "async function generateMagicPreview(" in script
    assert "async function openMagicCharacterDialog(" in script
    assert "magicEditor.markFullGeneration" in script
    assert "voice_profile: cloning ? (usesReference ? profileId : null) : profileId" in script
    assert "if (profile) restoreProfileGenerationSettings(profile)" in script
    assert "normalize_loudness: $('#normalize-loudness').checked" in script
    assert "useProfile(profile, 'clone', { editing: true })" in script
    assert "referenceAudio.clear()" in script
    assert "compactGeneratedReferenceText" in script
    assert "clone_mode: usesReference ? state.cloneMode : 'auto'" in script
    assert "function generatedVoiceRecipe(" in script
    assert "async function processDesignedVoice(" in script
    assert "requestPostProcessedAudio" in script
    assert "sample_text: payload.text" in script
    assert "design_reference_audio" in script
    assert "profile.tags" in script
    assert "portraitFile" in script
    assert "metadataOnly" in script
    assert "Update details" in script
    assert "async function loadProfileReference(" in script
    assert "sessionStorage.setItem(GPU_SESSION_KEY" in script


def test_ui_locale_catalogs_are_valid_and_english_covers_used_keys() -> None:
    static_dir = Path(runtime.__file__).parent / "standalone_ui" / "static"
    english = json.loads((static_dir / "locales" / "en.json").read_text(encoding="utf-8"))
    source = "\n".join(
        (
            (static_dir / "app.js").read_text(encoding="utf-8"),
            (static_dir / "magic-editor.js").read_text(encoding="utf-8"),
            (static_dir / "index.html").read_text(encoding="utf-8"),
        )
    )

    used_keys = set(re.findall(r"\bt\(\s*'([^']+)'\s*,", source))
    used_keys.update(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', source))
    assert used_keys <= english.keys()

    for locale in ("en", "nb", "pl", "ja", "zh", "es"):
        catalog = json.loads((static_dir / "locales" / f"{locale}.json").read_text(encoding="utf-8"))
        assert catalog
        assert set(catalog) <= set(english)
        if locale != "en":
            assert {key for key in english if not key.startswith("languages.")} <= set(catalog)
        assert all(isinstance(value, str) and "\ufffd" not in value for value in catalog.values())


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


@pytest.mark.parametrize("suffix", [".m4a", ".aac", ".webm"])
def test_reference_audio_normalization_produces_lossless_pcm_without_downmixing(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = tmp_path / f"reference{suffix}"
    source.write_bytes(encoded_reference_audio(tmp_path, suffix))

    normalized_path = Path(runtime.normalize_reference_audio(str(source)))
    try:
        info = sf.info(io.BytesIO(normalized_path.read_bytes()))
        assert normalized_path.suffix == ".wav"
        assert info.format == "WAV"
        assert info.subtype == "FLOAT"
        assert info.samplerate == 48_000
        assert info.channels == 2
    finally:
        normalized_path.unlink(missing_ok=True)


def test_m4a_clone_upload_is_normalized_and_removed(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_response(payload, route_name):
        path = Path(payload.ref_audio)
        info = sf.info(io.BytesIO(path.read_bytes()))
        observed.update(path=path, suffix=path.suffix, info=info, route=route_name)
        return StreamingResponse(io.BytesIO(b"audio"), media_type="audio/mpeg")

    request = {"text": "Clone this voice.", "ref_text": "Reference speech."}
    m4a_bytes = encoded_reference_audio(tmp_path, ".m4a")
    with patch.object(runtime, "stream_audio_response", side_effect=fake_response):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/generate-upload",
                data={"payload": json.dumps(request)},
                files={"reference_audio": ("reference.m4a", m4a_bytes, "audio/mp4")},
            )

    assert response.status_code == 200
    assert observed["suffix"] == ".wav"
    assert observed["info"].format == "WAV"
    assert observed["info"].subtype == "FLOAT"
    assert observed["route"] == "/tts/generate-upload"
    assert not observed["path"].exists()


def test_malformed_m4a_upload_is_rejected() -> None:
    with TestClient(runtime.app) as client:
        response = client.post(
            "/tts/generate-upload",
            data={"payload": json.dumps({"text": "Hello"})},
            files={"reference_audio": ("reference.m4a", b"not-an-audio-file", "audio/mp4")},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Unable to process reference audio:")


def test_reference_upload_limit_is_checked_before_transcoding() -> None:
    with patch.object(runtime, "MAX_REFERENCE_UPLOAD_BYTES", 4):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/generate-upload",
                data={"payload": json.dumps({"text": "Hello"})},
                files={"reference_audio": ("reference.m4a", b"too-large", "audio/mp4")},
            )

    assert response.status_code == 413
    assert response.json()["detail"] == "Reference audio exceeds the upload limit"


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

    def fake_encoder(chunks, output_format, sample_rate, *, normalize_loudness=False):
        observed.update(
            output_format=output_format,
            sample_rate=sample_rate,
            chunks=list(chunks),
            normalize_loudness=normalize_loudness,
        )
        yield b"first-"
        yield b"second"

    request = {"text": "Stream this.", "stream_format": "mp3", "normalize_loudness": True}
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
    assert observed["normalize_loudness"] is True
    assert response.headers["x-voxcpm-loudness-normalized"] == "true"
    assert not observed["path"].exists()


def test_loudness_normalization_uses_shared_ffmpeg_target() -> None:
    source = np.array([0.1, -0.1], dtype=np.float32)
    normalized_pcm = np.array([8192, -8192], dtype="<i2")
    completed = Mock(stdout=normalized_pcm.tobytes(), stderr=b"")

    with patch.object(runtime.subprocess, "run", return_value=completed) as run:
        result = runtime.normalize_audio_loudness(source, 24_000)

    command = run.call_args.args[0]
    assert command[command.index("-af") + 1] == "loudnorm=I=-16:TP=-1.5:LRA=11"
    assert command[command.index("-ar") + 1] == "24000"
    assert np.allclose(result, np.array([0.25, -0.25], dtype=np.float32))


def test_post_processing_filter_uses_bounded_voice_finishing_controls() -> None:
    payload = runtime.PostProcessingRequest(
        preset="custom",
        noise_reduction_db=4.5,
        bass_db=2.0,
        presence_db=-1.5,
        dynamics=40,
        normalize_loudness=True,
    )

    audio_filter = runtime.build_post_processing_filter(payload)

    assert audio_filter.startswith("highpass=f=50:p=2,afftdn=nr=4.5:nf=-50:tn=1:gs=6")
    assert "bass=f=110:t=q:w=0.7:g=2:p=2:r=f32" in audio_filter
    assert "treble=f=3500:t=q:w=0.7:g=-1.5:p=2:r=f32" in audio_filter
    assert "acompressor=" in audio_filter
    assert "mix=0.4" in audio_filter
    assert "deesser=i=0.12:m=0.35:f=0.5" in audio_filter
    assert "alimiter=limit=0.841:attack=5:release=50:level=disabled" in audio_filter
    assert audio_filter.endswith(runtime.LOUDNESS_NORMALIZATION_FILTER)


def test_ffmpeg_post_processing_produces_float_wav(tmp_path: Path) -> None:
    sample_rate = 48_000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = tmp_path / "source.wav"
    sf.write(source, 0.15 * np.sin(2 * np.pi * 180 * time_axis), sample_rate, subtype="FLOAT")

    processed = runtime.post_process_audio(
        str(source),
        runtime.PostProcessingRequest(
            preset="studio",
            noise_reduction_db=2,
            bass_db=1,
            presence_db=1,
            dynamics=35,
        ),
    )
    info = sf.info(io.BytesIO(processed))

    assert info.format in {"WAV", "WAVEX"}
    assert info.subtype == "FLOAT"
    assert info.samplerate == sample_rate
    assert info.channels == 1
    assert info.frames > 0


def test_signalsmith_post_processing_changes_pitch_and_speed(tmp_path: Path) -> None:
    sample_rate = 48_000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = tmp_path / "source.wav"
    sf.write(source, 0.15 * np.sin(2 * np.pi * 220 * time_axis), sample_rate, subtype="FLOAT")

    processed = runtime.post_process_audio(
        str(source),
        runtime.PostProcessingRequest(
            method="signalsmith",
            preset="custom",
            pitch_semitones=3,
            speed_factor=1.25,
            noise_reduction_db=0,
            bass_db=0,
            presence_db=0,
            dynamics=0,
            normalize_loudness=False,
        ),
    )
    audio, output_rate = sf.read(io.BytesIO(processed), dtype="float32")
    frequencies = np.fft.rfftfreq(audio.size, 1 / output_rate)
    peak_frequency = frequencies[int(np.argmax(np.abs(np.fft.rfft(audio * np.hanning(audio.size)))))]

    assert output_rate == sample_rate
    assert 0.78 <= audio.size / sample_rate <= 0.82
    assert 255 <= peak_frequency <= 268


def test_postprocess_upload_removes_source_and_returns_recipe_headers() -> None:
    observed: dict[str, object] = {}

    def fake_post_process(path: str, payload: runtime.PostProcessingRequest) -> bytes:
        source = Path(path)
        observed.update(path=source, exists=source.exists(), contents=source.read_bytes(), payload=payload)
        return b"processed-wave"

    options = {
        "method": "ffmpeg",
        "preset": "clean",
        "noise_reduction_db": 1,
        "bass_db": 0,
        "presence_db": 0.5,
        "dynamics": 15,
        "normalize_loudness": True,
    }
    with patch.object(runtime, "post_process_audio", side_effect=fake_post_process):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/postprocess-upload",
                data={"options": json.dumps(options)},
                files={"audio": ("generated.wav", b"source-wave", "audio/wav")},
            )

    assert response.status_code == 200
    assert response.content == b"processed-wave"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-voxcpm-post-processing"] == "ffmpeg"
    assert response.headers["x-voxcpm-post-processing-preset"] == "clean"
    assert observed["exists"] is True
    assert observed["contents"] == b"source-wave"
    assert observed["payload"].bass_db == 0
    assert not observed["path"].exists()


def test_postprocess_upload_rejects_out_of_range_controls() -> None:
    with TestClient(runtime.app) as client:
        response = client.post(
            "/tts/postprocess-upload",
            data={"options": json.dumps({"method": "ffmpeg", "preset": "custom", "bass_db": 20})},
            files={"audio": ("generated.wav", b"source-wave", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Invalid post-processing options:")


def test_defaults_enable_browser_loudness_normalization() -> None:
    with TestClient(runtime.app) as client:
        response = client.get("/tts/defaults")
        status = client.get("/tts/status")

    assert response.status_code == 200
    assert response.json()["normalize_loudness"] is True
    assert status.status_code == 200
    assert status.json()["post_processing"]["methods"] == ["ffmpeg", "signalsmith"]


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


def test_m4a_transcription_receives_normalized_audio(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_transcribe(path, language):
        audio_path = Path(path)
        observed.update(
            path=audio_path,
            suffix=audio_path.suffix,
            info=sf.info(io.BytesIO(audio_path.read_bytes())),
        )
        return "Reference transcript"

    m4a_bytes = encoded_reference_audio(tmp_path, ".m4a")
    with patch.object(runtime, "transcribe_reference_audio", side_effect=fake_transcribe):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/transcribe-upload",
                data={"language": "auto"},
                files={"reference_audio": ("reference.m4a", m4a_bytes, "audio/mp4")},
            )

    assert response.status_code == 200
    assert response.json()["text"] == "Reference transcript"
    assert observed["suffix"] == ".wav"
    assert observed["info"].format == "WAV"
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
                    "tags": json.dumps(["Warm", "Narrator", "warm"]),
                    "ref_text": "This is the original recording.",
                    "language": "English",
                    "recipe": json.dumps(
                        {
                            "design_source": "reference",
                            "clone_mode": "transcript",
                            "seed": 1234,
                            "randomize_seed": False,
                            "cfg_value": 2.5,
                            "inference_timesteps": 10,
                            "normalize": False,
                            "normalize_loudness": True,
                            "denoise": False,
                            "output_format": "wav",
                            "sample_text": "A complete voice design sample.",
                            "reference_text": "This is the original recording.",
                            "post_processing": {
                                "method": "signalsmith",
                                "preset": "custom",
                                "pitch_semitones": -2.5,
                                "speed_factor": 0.9,
                                "noise_reduction_db": 2,
                                "bass_db": 1,
                                "presence_db": 1,
                                "dynamics": 35,
                                "normalize_loudness": True,
                            },
                        }
                    ),
                },
                files={
                    "reference_audio": ("reference.wav", b"reference-bytes", "audio/wav"),
                    "design_reference_audio": ("original.wav", b"original-design-bytes", "audio/wav"),
                    "portrait": ("portrait.webp", TEST_PORTRAIT_WEBP, "image/webp"),
                },
            )
            listed = client.get("/tts/voice-profiles")
            audio = client.get("/tts/voice-profiles/studio-narrator/audio")
            design_audio = client.get("/tts/voice-profiles/studio-narrator/design-audio")
            portrait = client.get("/tts/voice-profiles/studio-narrator/portrait")

            payload = runtime.TTSRequest(text="A new sentence.", voice_profile="studio-narrator")
            kwargs = runtime.build_generate_kwargs(payload, object())

            deleted = client.delete("/tts/voice-profiles/studio-narrator")
            missing = client.get("/tts/voice-profiles/studio-narrator/audio")
            missing_design = client.get("/tts/voice-profiles/studio-narrator/design-audio")
            missing_portrait = client.get("/tts/voice-profiles/studio-narrator/portrait")

    assert created.status_code == 200
    assert created.json()["id"] == "studio-narrator"
    assert created.json()["profile_type"] == "cloned"
    assert created.json()["recipe"]["seed"] == 1234
    assert created.json()["recipe"]["clone_mode"] == "transcript"
    assert created.json()["recipe"]["sample_text"] == "A complete voice design sample."
    assert created.json()["recipe"]["post_processing"]["method"] == "signalsmith"
    assert created.json()["recipe"]["post_processing"]["pitch_semitones"] == -2.5
    assert created.json()["recipe"]["post_processing"]["speed_factor"] == 0.9
    assert created.json()["recipe"]["post_processing"]["dynamics"] == 35
    assert created.json()["has_design_audio"] is True
    assert created.json()["has_portrait"] is True
    assert created.json()["tags"] == ["Warm", "Narrator"]
    assert listed.json()["count"] == 1
    assert listed.headers["cache-control"] == "no-store"
    assert audio.content == b"reference-bytes"
    assert design_audio.content == b"original-design-bytes"
    assert portrait.content.startswith(b"RIFF") and portrait.content[8:12] == b"WEBP"
    assert portrait.headers["content-type"] == "image/webp"
    assert Path(kwargs["reference_wav_path"]).parent == tmp_path
    assert kwargs["prompt_wav_path"] == kwargs["reference_wav_path"]
    assert kwargs["prompt_text"] == "This is the original recording."
    assert 0 <= kwargs["seed"] <= runtime.MAX_RANDOM_SEED
    assert deleted.json() == {"deleted": "studio-narrator"}
    assert missing.status_code == 404
    assert missing_design.status_code == 404
    assert missing_portrait.status_code == 404


def test_saved_clone_profile_can_switch_from_transcript_to_directed_reference_mode(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            created = client.post(
                "/tts/voice-profiles",
                data={"name": "Dialogue Voice", "profile_type": "cloned", "ref_text": "Original words."},
                files={"reference_audio": ("reference.wav", b"reference-bytes", "audio/wav")},
            )

        transcript_kwargs = runtime.build_generate_kwargs(
            runtime.TTSRequest(text="A new sentence.", voice_profile="dialogue-voice"),
            object(),
        )
        directed_kwargs = runtime.build_generate_kwargs(
            runtime.TTSRequest(
                text="A new sentence.",
                voice_profile="dialogue-voice",
                clone_mode="reference",
                control="Calm but authoritative",
            ),
            object(),
        )

    assert created.status_code == 200
    assert transcript_kwargs["prompt_text"] == "Original words."
    assert transcript_kwargs["prompt_wav_path"] == transcript_kwargs["reference_wav_path"]
    assert directed_kwargs["prompt_text"] is None
    assert directed_kwargs["prompt_wav_path"] is None
    assert directed_kwargs["text"] == "(Calm but authoritative)A new sentence."


def test_saved_clone_profile_can_be_edited_without_replacing_audio(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            created = client.post(
                "/tts/voice-profiles",
                data={
                    "name": "Studio Voice",
                    "profile_type": "cloned",
                    "ref_text": "Original.",
                    "tags": json.dumps(["Studio", "English"]),
                    "recipe": json.dumps({"design_source": "reference", "seed": 99}),
                },
                files={
                    "reference_audio": ("reference.wav", b"reference-bytes", "audio/wav"),
                    "design_reference_audio": ("design.wav", b"design-bytes", "audio/wav"),
                    "portrait": ("portrait.webp", TEST_PORTRAIT_WEBP, "image/webp"),
                },
            )
            created_record = runtime.load_voice_profiles(tmp_path)["studio-voice"]
            updated = client.put(
                "/tts/voice-profiles/studio-voice",
                data={
                    "description": "Updated studio voice",
                    "ref_text": "Updated transcript.",
                    "language": "English",
                },
            )
            audio = client.get("/tts/voice-profiles/studio-voice/audio")
            design_audio = client.get("/tts/voice-profiles/studio-voice/design-audio")
            portrait = client.get("/tts/voice-profiles/studio-voice/portrait")
            tags_only = client.put(
                "/tts/voice-profiles/studio-voice",
                data={"tags": json.dumps(["Updated", "Metadata Only"])},
            )
            tags_only_record = runtime.load_voice_profiles(tmp_path)["studio-voice"]
            cleared = client.put(
                "/tts/voice-profiles/studio-voice",
                data={
                    "description": "Updated studio voice",
                    "ref_text": "Updated transcript.",
                    "language": "English",
                    "tags": "[]",
                    "clear_portrait": "true",
                },
            )
            missing_portrait = client.get("/tts/voice-profiles/studio-voice/portrait")

    assert created.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["description"] == "Updated studio voice"
    assert updated.json()["ref_text"] == "Updated transcript."
    assert updated.json()["recipe"] == {"design_source": "reference", "seed": 99}
    assert updated.json()["tags"] == ["Studio", "English"]
    assert audio.content == b"reference-bytes"
    assert design_audio.content == b"design-bytes"
    assert portrait.content.startswith(b"RIFF") and portrait.content[8:12] == b"WEBP"
    assert tags_only.json()["tags"] == ["Updated", "Metadata Only"]
    assert tags_only.json()["description"] == "Updated studio voice"
    assert tags_only.json()["ref_text"] == "Updated transcript."
    assert tags_only.json()["recipe"] == {"design_source": "reference", "seed": 99}
    assert tags_only_record["audio_file"] == created_record["audio_file"]
    assert tags_only_record["design_audio_file"] == created_record["design_audio_file"]
    assert tags_only_record["portrait_file"] == created_record["portrait_file"]
    assert tags_only_record["created_at"] == created_record["created_at"]
    assert cleared.json()["has_portrait"] is False
    assert cleared.json()["tags"] == []
    assert missing_portrait.status_code == 404


def test_saved_voice_can_be_replaced_with_a_refined_design(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            created = client.post(
                "/tts/voice-profiles",
                data={"name": "Bob", "profile_type": "cloned", "ref_text": "Old words."},
                files={
                    "reference_audio": ("old.wav", b"old-production", "audio/wav"),
                    "design_reference_audio": ("old-design.wav", b"old-design", "audio/wav"),
                },
            )
            updated = client.put(
                "/tts/voice-profiles/bob",
                data={
                    "profile_type": "cloned",
                    "description": "Refined Bob",
                    "tags": json.dumps(["Captain", "Polish"]),
                    "ref_text": "New compact words.",
                    "control": "Warm and authoritative",
                    "language": "English",
                    "recipe": json.dumps(
                        {
                            "design_source": "reference",
                            "clone_mode": "reference",
                            "seed": 4242,
                            "randomize_seed": False,
                            "sample_text": "The exact text used to refine Bob.",
                            "reference_text": "",
                        }
                    ),
                },
                files={
                    "reference_audio": ("new.wav", b"new-production", "audio/wav"),
                    "design_reference_audio": ("new-design.wav", b"new-design", "audio/wav"),
                    "portrait": ("new.webp", TEST_PORTRAIT_WEBP, "image/webp"),
                },
            )
            audio = client.get("/tts/voice-profiles/bob/audio")
            design_audio = client.get("/tts/voice-profiles/bob/design-audio")
            portrait = client.get("/tts/voice-profiles/bob/portrait")

    assert created.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["description"] == "Refined Bob"
    assert updated.json()["control"] == "Warm and authoritative"
    assert updated.json()["tags"] == ["Captain", "Polish"]
    assert updated.json()["recipe"]["sample_text"] == "The exact text used to refine Bob."
    assert updated.json()["recipe"]["seed"] == 4242
    assert audio.content == b"new-production"
    assert design_audio.content == b"new-design"
    assert portrait.content.startswith(b"RIFF") and portrait.content[8:12] == b"WEBP"


def test_voice_profile_rejects_spoofed_portrait(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/voice-profiles",
                data={"name": "Spoofed", "profile_type": "cloned"},
                files={
                    "reference_audio": ("reference.wav", b"reference-bytes", "audio/wav"),
                    "portrait": ("portrait.png", b"not-an-image", "image/png"),
                },
            )

    assert response.status_code == 400
    assert "contents" in response.json()["detail"].lower()


def test_voice_profile_rejects_invalid_design_recipe(tmp_path: Path) -> None:
    with patch.object(runtime, "VOICE_PROFILE_DIR", tmp_path):
        with TestClient(runtime.app) as client:
            response = client.post(
                "/tts/voice-profiles",
                data={
                    "name": "Invalid Recipe",
                    "profile_type": "designed",
                    "control": "A warm narrator",
                    "recipe": json.dumps({"design_source": "unknown"}),
                },
            )

    assert response.status_code == 400
    assert "design source" in response.json()["detail"].lower()


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
