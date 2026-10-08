from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROFILE_LOCK = threading.RLock()
PROFILE_TYPES = {"cloned", "designed"}


def normalize_profile_name(name: str | None) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-_")
    if not value:
        raise ValueError("Voice name must contain at least one letter or number.")
    if len(value) > 48:
        raise ValueError("Voice name must be 48 characters or fewer.")
    return value


def _saved_audio_path(profile_dir: Path, audio_file: str) -> Path | None:
    if not audio_file:
        return None
    root = profile_dir.resolve()
    candidate = (root / Path(audio_file).name).resolve()
    if candidate.parent != root or not candidate.is_file():
        return None
    return candidate


def _write_index(profile_dir: Path, profiles: dict[str, dict[str, Any]]) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    index_path = profile_dir / "profiles.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=profile_dir,
            prefix=".profiles.",
            suffix=".tmp",
        ) as output:
            temporary_path = Path(output.name)
            json.dump(profiles, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, index_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_voice_profiles(profile_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = profile_dir / "profiles.json"
    if not index_path.exists():
        return {}
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for raw_name, raw_profile in raw.items():
        if not isinstance(raw_profile, dict):
            continue
        try:
            name = normalize_profile_name(str(raw_name))
        except ValueError:
            continue
        profile_type = str(raw_profile.get("profile_type") or "cloned")
        if profile_type not in PROFILE_TYPES:
            continue
        audio_file = Path(str(raw_profile.get("audio_file") or "")).name
        if profile_type == "cloned" and _saved_audio_path(profile_dir, audio_file) is None:
            continue
        profiles[name] = {
            "profile_type": profile_type,
            "audio_file": audio_file,
            "ref_text": str(raw_profile.get("ref_text") or "").strip(),
            "control": str(raw_profile.get("control") or "").strip(),
            "description": str(raw_profile.get("description") or "").strip(),
            "language": str(raw_profile.get("language") or "").strip(),
            "created_at": str(raw_profile.get("created_at") or ""),
        }
    return profiles


def save_voice_profile(
    profile_dir: Path,
    *,
    name: str,
    profile_type: str,
    source_audio: str | None = None,
    ref_text: str | None = None,
    control: str | None = None,
    description: str | None = None,
    language: str | None = None,
    overwrite: bool = True,
) -> tuple[str, dict[str, Any]]:
    profile_name = normalize_profile_name(name)
    normalized_type = (profile_type or "").strip().lower()
    if normalized_type not in PROFILE_TYPES:
        raise ValueError("Voice type must be 'cloned' or 'designed'.")
    normalized_control = (control or "").strip()
    normalized_description = (description or "").strip()
    if len(normalized_description) > 240:
        raise ValueError("Voice description must be 240 characters or fewer.")
    if normalized_type == "designed" and not normalized_control:
        raise ValueError("Enter a voice description before saving a designed voice.")
    source = Path(source_audio).resolve() if source_audio else None
    if normalized_type == "cloned" and (source is None or not source.is_file()):
        raise ValueError("Upload a reference audio sample before saving a cloned voice.")

    with PROFILE_LOCK:
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiles = load_voice_profiles(profile_dir)
        if not overwrite and profile_name in profiles:
            raise ValueError(f"Voice profile '{profile_name}' already exists.")
        previous_audio = (profiles.get(profile_name) or {}).get("audio_file")
        destination: Path | None = None
        try:
            if source is not None:
                suffix = source.suffix.lower() or ".wav"
                destination = profile_dir / f"voice-{profile_name}-{os.urandom(6).hex()}{suffix}"
                shutil.copyfile(source, destination)
            profile = {
                "profile_type": normalized_type,
                "audio_file": destination.name if destination else "",
                "ref_text": (ref_text or "").strip(),
                "control": normalized_control,
                "description": normalized_description,
                "language": (language or "").strip(),
                "created_at": datetime.now(UTC).isoformat(),
            }
            profiles[profile_name] = profile
            _write_index(profile_dir, profiles)
        except Exception:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise

        if previous_audio and previous_audio != profile["audio_file"]:
            (profile_dir / Path(previous_audio).name).unlink(missing_ok=True)
        return profile_name, profile


def update_voice_profile(
    profile_dir: Path,
    *,
    name: str,
    description: str | None = None,
    ref_text: str | None = None,
    control: str | None = None,
    language: str | None = None,
    source_audio: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Update a profile while retaining an existing clone sample when omitted."""

    profile_name = normalize_profile_name(name)
    with PROFILE_LOCK:
        _, existing = resolve_voice_profile(profile_dir, profile_name)
        retained_audio = source_audio
        if existing["profile_type"] == "cloned" and not retained_audio:
            retained_audio = existing.get("ref_audio") or None
        return save_voice_profile(
            profile_dir,
            name=profile_name,
            profile_type=existing["profile_type"],
            source_audio=retained_audio,
            ref_text=ref_text,
            control=control,
            description=description,
            language=language,
            overwrite=True,
        )


def delete_voice_profile(profile_dir: Path, name: str) -> str:
    profile_name = normalize_profile_name(name)
    with PROFILE_LOCK:
        profiles = load_voice_profiles(profile_dir)
        profile = profiles.pop(profile_name, None)
        if profile is None:
            raise ValueError(f"Voice profile '{profile_name}' does not exist.")
        _write_index(profile_dir, profiles)
        audio_file = Path(str(profile.get("audio_file") or "")).name
        if audio_file:
            (profile_dir / audio_file).unlink(missing_ok=True)
    return profile_name


def resolve_voice_profile(profile_dir: Path, name: str) -> tuple[str, dict[str, Any]]:
    profile_name = normalize_profile_name(name)
    with PROFILE_LOCK:
        profile = load_voice_profiles(profile_dir).get(profile_name)
    if profile is None:
        raise ValueError(f"Voice profile '{profile_name}' does not exist.")
    resolved = dict(profile)
    audio_file = Path(str(profile.get("audio_file") or "")).name
    audio_path = _saved_audio_path(profile_dir, audio_file)
    if profile["profile_type"] == "cloned" and audio_path is None:
        raise ValueError(f"Voice profile '{profile_name}' has no readable reference audio.")
    resolved["ref_audio"] = str(audio_path) if audio_path else ""
    return profile_name, resolved
