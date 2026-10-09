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
PROFILE_DESIGN_SOURCES = {"reference", "direction"}
PROFILE_CLONE_MODES = {"reference", "transcript"}
PROFILE_OUTPUT_FORMATS = {"wav", "mp3", "flac", "ogg", "opus", "aac"}
PROFILE_POST_PROCESSING_METHODS = {"ffmpeg", "signalsmith"}
PROFILE_POST_PROCESSING_PRESETS = {"clean", "studio", "custom"}
MAX_PROFILE_SAMPLE_CHARACTERS = 50_000
MAX_PROFILE_TAGS = 12
MAX_PROFILE_TAG_CHARACTERS = 32


def normalize_profile_tags(tags: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)):
        raise ValueError("Voice profile tags must be a JSON array.")
    if len(tags) > MAX_PROFILE_TAGS:
        raise ValueError(f"Voice profiles support at most {MAX_PROFILE_TAGS} tags.")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = re.sub(r"\s+", " ", str(raw_tag or "").strip())
        if not tag:
            continue
        if len(tag) > MAX_PROFILE_TAG_CHARACTERS:
            raise ValueError(f"Voice profile tags must be {MAX_PROFILE_TAG_CHARACTERS} characters or fewer.")
        identity = tag.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(tag)
    return normalized


def normalize_profile_recipe(recipe: dict[str, Any] | None) -> dict[str, Any]:
    if not recipe:
        return {}
    if not isinstance(recipe, dict):
        raise ValueError("Voice design recipe must be a JSON object.")

    normalized: dict[str, Any] = {}
    design_source = str(recipe.get("design_source") or "").strip().lower()
    if design_source:
        if design_source not in PROFILE_DESIGN_SOURCES:
            raise ValueError("Voice design source must be 'reference' or 'direction'.")
        normalized["design_source"] = design_source

    clone_mode = str(recipe.get("clone_mode") or "").strip().lower()
    if clone_mode:
        if clone_mode not in PROFILE_CLONE_MODES:
            raise ValueError("Voice clone mode must be 'reference' or 'transcript'.")
        normalized["clone_mode"] = clone_mode

    output_format = str(recipe.get("output_format") or "").strip().lower()
    if output_format:
        if output_format not in PROFILE_OUTPUT_FORMATS:
            raise ValueError("Voice recipe output format is unsupported.")
        normalized["output_format"] = output_format

    for key in ("normalize", "normalize_loudness", "denoise", "randomize_seed"):
        if key in recipe:
            if not isinstance(recipe[key], bool):
                raise ValueError(f"Voice recipe {key} must be true or false.")
            normalized[key] = recipe[key]

    if "seed" in recipe:
        seed = int(recipe["seed"])
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Voice recipe seed must be between 0 and 4294967295.")
        normalized["seed"] = seed

    if "cfg_value" in recipe:
        cfg_value = float(recipe["cfg_value"])
        if not 0.1 <= cfg_value <= 10.0:
            raise ValueError("Voice recipe guidance must be between 0.1 and 10.0.")
        normalized["cfg_value"] = cfg_value

    if "inference_timesteps" in recipe:
        inference_timesteps = int(recipe["inference_timesteps"])
        if not 1 <= inference_timesteps <= 100:
            raise ValueError("Voice recipe inference steps must be between 1 and 100.")
        normalized["inference_timesteps"] = inference_timesteps

    for key, label in (("sample_text", "design sample text"), ("reference_text", "reference transcript")):
        if key not in recipe:
            continue
        value = str(recipe[key] or "").strip()
        if len(value) > MAX_PROFILE_SAMPLE_CHARACTERS:
            raise ValueError(f"Voice recipe {label} must be {MAX_PROFILE_SAMPLE_CHARACTERS} characters or fewer.")
        normalized[key] = value

    post_processing = recipe.get("post_processing")
    if post_processing is not None:
        if not isinstance(post_processing, dict):
            raise ValueError("Voice recipe post-processing must be a JSON object.")
        method = str(post_processing.get("method") or "").strip().lower()
        preset = str(post_processing.get("preset") or "").strip().lower()
        if method not in PROFILE_POST_PROCESSING_METHODS:
            raise ValueError("Voice recipe post-processing method is unsupported.")
        if preset not in PROFILE_POST_PROCESSING_PRESETS:
            raise ValueError("Voice recipe post-processing preset is unsupported.")
        normalized_post_processing: dict[str, Any] = {"method": method, "preset": preset}
        for key, default, minimum, maximum in (
            ("pitch_semitones", 0.0, -6.0, 6.0),
            ("speed_factor", 1.0, 0.75, 1.25),
        ):
            value = float(post_processing.get(key, default))
            if not minimum <= value <= maximum:
                raise ValueError(f"Voice recipe {key} must be between {minimum:g} and {maximum:g}.")
            normalized_post_processing[key] = value
        for key, minimum, maximum in (
            ("noise_reduction_db", 0.0, 12.0),
            ("bass_db", -6.0, 6.0),
            ("presence_db", -6.0, 6.0),
        ):
            value = float(post_processing.get(key, 0.0))
            if not minimum <= value <= maximum:
                raise ValueError(f"Voice recipe {key} must be between {minimum:g} and {maximum:g}.")
            normalized_post_processing[key] = value
        dynamics = int(post_processing.get("dynamics", 0))
        if not 0 <= dynamics <= 100:
            raise ValueError("Voice recipe dynamics must be between 0 and 100.")
        normalized_post_processing["dynamics"] = dynamics
        normalize_loudness = post_processing.get("normalize_loudness", True)
        if not isinstance(normalize_loudness, bool):
            raise ValueError("Voice recipe post-processing normalize_loudness must be true or false.")
        normalized_post_processing["normalize_loudness"] = normalize_loudness
        normalized["post_processing"] = normalized_post_processing

    return normalized


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
        design_audio_file = Path(str(raw_profile.get("design_audio_file") or "")).name
        portrait_file = Path(str(raw_profile.get("portrait_file") or "")).name
        if profile_type == "cloned" and _saved_audio_path(profile_dir, audio_file) is None:
            continue
        if design_audio_file and _saved_audio_path(profile_dir, design_audio_file) is None:
            design_audio_file = ""
        if portrait_file and _saved_audio_path(profile_dir, portrait_file) is None:
            portrait_file = ""
        try:
            recipe = normalize_profile_recipe(raw_profile.get("recipe"))
        except (TypeError, ValueError):
            recipe = {}
        try:
            tags = normalize_profile_tags(raw_profile.get("tags"))
        except (TypeError, ValueError):
            tags = []
        profiles[name] = {
            "profile_type": profile_type,
            "audio_file": audio_file,
            "design_audio_file": design_audio_file,
            "portrait_file": portrait_file,
            "ref_text": str(raw_profile.get("ref_text") or "").strip(),
            "control": str(raw_profile.get("control") or "").strip(),
            "description": str(raw_profile.get("description") or "").strip(),
            "tags": tags,
            "language": str(raw_profile.get("language") or "").strip(),
            "recipe": recipe,
            "created_at": str(raw_profile.get("created_at") or ""),
        }
    return profiles


def save_voice_profile(
    profile_dir: Path,
    *,
    name: str,
    profile_type: str,
    source_audio: str | None = None,
    design_source_audio: str | None = None,
    portrait_source: str | None = None,
    ref_text: str | None = None,
    control: str | None = None,
    description: str | None = None,
    tags: list[Any] | tuple[Any, ...] | None = None,
    language: str | None = None,
    recipe: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> tuple[str, dict[str, Any]]:
    profile_name = normalize_profile_name(name)
    normalized_type = (profile_type or "").strip().lower()
    if normalized_type not in PROFILE_TYPES:
        raise ValueError("Voice type must be 'cloned' or 'designed'.")
    normalized_control = (control or "").strip()
    normalized_description = (description or "").strip()
    normalized_tags = normalize_profile_tags(tags)
    normalized_recipe = normalize_profile_recipe(recipe)
    if len(normalized_description) > 240:
        raise ValueError("Voice description must be 240 characters or fewer.")
    if normalized_type == "designed" and not normalized_control:
        raise ValueError("Enter a voice description before saving a designed voice.")
    source = Path(source_audio).resolve() if source_audio else None
    design_source = Path(design_source_audio).resolve() if design_source_audio else None
    portrait = Path(portrait_source).resolve() if portrait_source else None
    if normalized_type == "cloned" and (source is None or not source.is_file()):
        raise ValueError("Upload a reference audio sample before saving a cloned voice.")
    if design_source is not None and not design_source.is_file():
        raise ValueError("Voice design reference audio does not exist.")
    if portrait is not None and not portrait.is_file():
        raise ValueError("Voice portrait does not exist.")

    with PROFILE_LOCK:
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiles = load_voice_profiles(profile_dir)
        if not overwrite and profile_name in profiles:
            raise ValueError(f"Voice profile '{profile_name}' already exists.")
        previous_audio = (profiles.get(profile_name) or {}).get("audio_file")
        previous_design_audio = (profiles.get(profile_name) or {}).get("design_audio_file")
        previous_portrait = (profiles.get(profile_name) or {}).get("portrait_file")
        destination: Path | None = None
        design_destination: Path | None = None
        portrait_destination: Path | None = None
        try:
            if source is not None:
                suffix = source.suffix.lower() or ".wav"
                destination = profile_dir / f"voice-{profile_name}-{os.urandom(6).hex()}{suffix}"
                shutil.copyfile(source, destination)
            if design_source is not None:
                suffix = design_source.suffix.lower() or ".wav"
                design_destination = profile_dir / f"design-{profile_name}-{os.urandom(6).hex()}{suffix}"
                shutil.copyfile(design_source, design_destination)
            if portrait is not None:
                suffix = portrait.suffix.lower() or ".webp"
                portrait_destination = profile_dir / f"portrait-{profile_name}-{os.urandom(6).hex()}{suffix}"
                shutil.copyfile(portrait, portrait_destination)
            profile = {
                "profile_type": normalized_type,
                "audio_file": destination.name if destination else "",
                "design_audio_file": design_destination.name if design_destination else "",
                "portrait_file": portrait_destination.name if portrait_destination else "",
                "ref_text": (ref_text or "").strip(),
                "control": normalized_control,
                "description": normalized_description,
                "tags": normalized_tags,
                "language": (language or "").strip(),
                "recipe": normalized_recipe,
                "created_at": datetime.now(UTC).isoformat(),
            }
            profiles[profile_name] = profile
            _write_index(profile_dir, profiles)
        except Exception:
            if destination is not None:
                destination.unlink(missing_ok=True)
            if design_destination is not None:
                design_destination.unlink(missing_ok=True)
            if portrait_destination is not None:
                portrait_destination.unlink(missing_ok=True)
            raise

        if previous_audio and previous_audio != profile["audio_file"]:
            (profile_dir / Path(previous_audio).name).unlink(missing_ok=True)
        if previous_design_audio and previous_design_audio != profile["design_audio_file"]:
            (profile_dir / Path(previous_design_audio).name).unlink(missing_ok=True)
        if previous_portrait and previous_portrait != profile["portrait_file"]:
            (profile_dir / Path(previous_portrait).name).unlink(missing_ok=True)
        return profile_name, profile


def update_voice_profile(
    profile_dir: Path,
    *,
    name: str,
    description: str | None = None,
    tags: list[Any] | tuple[Any, ...] | None = None,
    ref_text: str | None = None,
    control: str | None = None,
    language: str | None = None,
    recipe: dict[str, Any] | None = None,
    source_audio: str | None = None,
    design_source_audio: str | None = None,
    clear_design_audio: bool = False,
    portrait_source: str | None = None,
    clear_portrait: bool = False,
    profile_type: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Update only supplied fields and keep untouched profile assets in place."""

    profile_name = normalize_profile_name(name)
    with PROFILE_LOCK:
        profiles = load_voice_profiles(profile_dir)
        existing = profiles.get(profile_name)
        if existing is None:
            raise ValueError(f"Voice profile '{profile_name}' does not exist.")

        normalized_type = (profile_type or existing["profile_type"]).strip().lower()
        if normalized_type not in PROFILE_TYPES:
            raise ValueError("Voice type must be 'cloned' or 'designed'.")

        updated = dict(existing)
        updated["profile_type"] = normalized_type
        if description is not None:
            normalized_description = description.strip()
            if len(normalized_description) > 240:
                raise ValueError("Voice description must be 240 characters or fewer.")
            updated["description"] = normalized_description
        if tags is not None:
            updated["tags"] = normalize_profile_tags(tags)
        if ref_text is not None:
            updated["ref_text"] = ref_text.strip()
        if control is not None:
            updated["control"] = control.strip()
        if language is not None:
            updated["language"] = language.strip()
        if recipe is not None:
            updated["recipe"] = normalize_profile_recipe(recipe)
        if normalized_type == "designed" and not updated.get("control"):
            raise ValueError("Enter a voice description before saving a designed voice.")

        created_files: list[Path] = []
        replaced_files: list[str] = []

        def replace_asset(source_value: str, key: str, prefix: str, fallback_suffix: str) -> None:
            source = Path(source_value).resolve()
            if not source.is_file():
                raise ValueError(f"Voice profile {prefix} source does not exist.")
            suffix = source.suffix.lower() or fallback_suffix
            destination = profile_dir / f"{prefix}-{profile_name}-{os.urandom(6).hex()}{suffix}"
            shutil.copyfile(source, destination)
            created_files.append(destination)
            previous = str(updated.get(key) or "")
            updated[key] = destination.name
            if previous and previous != destination.name:
                replaced_files.append(previous)

        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            if source_audio:
                replace_asset(source_audio, "audio_file", "voice", ".wav")
            if design_source_audio:
                replace_asset(design_source_audio, "design_audio_file", "design", ".wav")
            elif clear_design_audio:
                previous = str(updated.get("design_audio_file") or "")
                updated["design_audio_file"] = ""
                if previous:
                    replaced_files.append(previous)
            if portrait_source:
                replace_asset(portrait_source, "portrait_file", "portrait", ".webp")
            elif clear_portrait:
                previous = str(updated.get("portrait_file") or "")
                updated["portrait_file"] = ""
                if previous:
                    replaced_files.append(previous)

            if normalized_type == "cloned" and _saved_audio_path(profile_dir, str(updated.get("audio_file") or "")) is None:
                raise ValueError("Upload a reference audio sample before saving a cloned voice.")

            profiles[profile_name] = updated
            _write_index(profile_dir, profiles)
        except Exception:
            for path in created_files:
                path.unlink(missing_ok=True)
            raise

        for previous in replaced_files:
            (profile_dir / Path(previous).name).unlink(missing_ok=True)
        return profile_name, updated


def delete_voice_profile(profile_dir: Path, name: str) -> str:
    profile_name = normalize_profile_name(name)
    with PROFILE_LOCK:
        profiles = load_voice_profiles(profile_dir)
        profile = profiles.pop(profile_name, None)
        if profile is None:
            raise ValueError(f"Voice profile '{profile_name}' does not exist.")
        _write_index(profile_dir, profiles)
        for key in ("audio_file", "design_audio_file", "portrait_file"):
            audio_file = Path(str(profile.get(key) or "")).name
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
    design_audio_file = Path(str(profile.get("design_audio_file") or "")).name
    design_audio_path = _saved_audio_path(profile_dir, design_audio_file)
    portrait_file = Path(str(profile.get("portrait_file") or "")).name
    portrait_path = _saved_audio_path(profile_dir, portrait_file)
    if profile["profile_type"] == "cloned" and audio_path is None:
        raise ValueError(f"Voice profile '{profile_name}' has no readable reference audio.")
    resolved["ref_audio"] = str(audio_path) if audio_path else ""
    resolved["design_ref_audio"] = str(design_audio_path) if design_audio_path else ""
    resolved["portrait_path"] = str(portrait_path) if portrait_path else ""
    return profile_name, resolved
