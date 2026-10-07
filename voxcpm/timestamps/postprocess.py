from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec

from .base import TimestampLevel


def timestamp_backend_available(backend: str = "stable-ts") -> bool:
    if backend != "stable-ts":
        return False
    return find_spec("stable_whisper") is not None


@lru_cache(maxsize=4)
def _stable_ts_aligner(model_name: str, device: str | None, language: str | None):
    from .stable_ts import StableTSAligner

    return StableTSAligner(model_name=model_name, device=device, language=language)


def align_audio_file(
    *,
    audio_path: str,
    text: str,
    sample_rate: int | None = None,
    backend: str = "stable-ts",
    level: TimestampLevel = "word",
    model_name: str = "base",
    device: str | None = None,
    language: str | None = None,
) -> dict:
    if backend != "stable-ts":
        raise ValueError(f"Unsupported timestamp backend: {backend}")

    return _stable_ts_aligner(model_name, device, language).align(
        audio_path=audio_path,
        text=text,
        sample_rate=sample_rate,
        level=level,
    ).to_dict()
