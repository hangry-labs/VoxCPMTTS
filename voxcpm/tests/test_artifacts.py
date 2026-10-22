from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from voxcpm.artifacts import ArtifactExpiredError, ArtifactNotFoundError, ArtifactStore


def create_artifact(store: ArtifactStore, *, now: float = 1000):
    return store.create(
        b"audio",
        extension="mp3",
        ttl_seconds=300,
        format="mp3",
        mime_type="audio/mpeg",
        duration_seconds=1.25,
        sample_rate=24_000,
        seed=42,
        voice_mode="random",
        voice="auto",
        language="auto",
        now=now,
    )


def test_artifact_metadata_never_contains_input_text_or_audio() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(directory)
        artifact = create_artifact(store)
        resolved, path = store.resolve(artifact.token, now=1200)
        metadata = json.loads((Path(directory) / f"{artifact.token}.json").read_text(encoding="utf-8"))

        assert path.read_bytes() == b"audio"
        assert resolved.size_bytes == 5
        assert "text" not in metadata
        assert "audio" not in metadata


def test_expired_artifact_is_removed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(directory)
        artifact = create_artifact(store)
        with pytest.raises(ArtifactExpiredError, match="expired"):
            store.resolve(artifact.token, now=1300)
        assert not any(Path(directory).iterdir())


def test_invalid_token_cannot_escape_output_directory() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(directory)
        with pytest.raises(ArtifactNotFoundError, match="not found"):
            store.resolve("../settings", now=1000)
