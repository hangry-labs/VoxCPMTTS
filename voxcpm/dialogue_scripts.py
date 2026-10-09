from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_LOCK = threading.RLock()
MAX_SCRIPT_NAME_CHARACTERS = 80
MAX_SCRIPT_DESCRIPTION_CHARACTERS = 240
MAX_SCRIPT_DOCUMENT_BYTES = 1_000_000
MAX_SCRIPT_TAGS = 12
MAX_SCRIPT_TAG_CHARACTERS = 32


def normalize_script_name(name: str | None) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-_")
    if not value:
        raise ValueError("Script name must contain at least one letter or number.")
    if len(value) > MAX_SCRIPT_NAME_CHARACTERS:
        raise ValueError(f"Script name must be {MAX_SCRIPT_NAME_CHARACTERS} characters or fewer.")
    return value


def normalize_script_document(document: str | None) -> str:
    value = (document or "").strip()
    if not value:
        raise ValueError("SSML-H document must not be empty.")
    if len(value.encode("utf-8")) > MAX_SCRIPT_DOCUMENT_BYTES:
        raise ValueError("SSML-H document exceeds the 1 MB storage limit.")
    return f"{value}\n"


def normalize_script_tags(tags: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)):
        raise ValueError("Dialogue script tags must be a JSON array.")
    if len(tags) > MAX_SCRIPT_TAGS:
        raise ValueError(f"Dialogue scripts support at most {MAX_SCRIPT_TAGS} tags.")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = re.sub(r"\s+", " ", str(raw_tag or "").strip())
        if not tag:
            continue
        if len(tag) > MAX_SCRIPT_TAG_CHARACTERS:
            raise ValueError(f"Dialogue script tags must be {MAX_SCRIPT_TAG_CHARACTERS} characters or fewer.")
        identity = tag.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(tag)
    return normalized


def _write_index(script_dir: Path, scripts: dict[str, dict[str, Any]]) -> None:
    script_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=script_dir,
            prefix=".scripts.",
            suffix=".tmp",
        ) as output:
            temporary_path = Path(output.name)
            json.dump(scripts, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, script_dir / "scripts.json")
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _document_path(script_dir: Path, filename: str) -> Path | None:
    if not filename:
        return None
    root = script_dir.resolve()
    candidate = (root / Path(filename).name).resolve()
    if candidate.parent != root or not candidate.is_file():
        return None
    return candidate


def load_dialogue_scripts(script_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = script_dir / "scripts.json"
    if not index_path.exists():
        return {}
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    scripts: dict[str, dict[str, Any]] = {}
    for raw_name, raw_script in raw.items():
        if not isinstance(raw_script, dict):
            continue
        try:
            name = normalize_script_name(str(raw_name))
        except ValueError:
            continue
        filename = Path(str(raw_script.get("document_file") or "")).name
        if _document_path(script_dir, filename) is None:
            continue
        try:
            turns = max(0, int(raw_script.get("turns") or 0))
            words = max(0, int(raw_script.get("words") or 0))
        except (TypeError, ValueError):
            turns = 0
            words = 0
        try:
            tags = normalize_script_tags(raw_script.get("tags"))
        except ValueError:
            tags = []
        scripts[name] = {
            "title": str(raw_script.get("title") or name).strip()[:MAX_SCRIPT_NAME_CHARACTERS],
            "description": str(raw_script.get("description") or "").strip()[:MAX_SCRIPT_DESCRIPTION_CHARACTERS],
            "tags": tags,
            "document_file": filename,
            "turns": turns,
            "words": words,
            "created_at": str(raw_script.get("created_at") or ""),
            "updated_at": str(raw_script.get("updated_at") or ""),
        }
    return scripts


def save_dialogue_script(
    script_dir: Path,
    *,
    name: str,
    document: str,
    description: str = "",
    tags: list[Any] | tuple[Any, ...] | None = None,
    turns: int = 0,
    words: int = 0,
    overwrite: bool = False,
) -> tuple[str, dict[str, Any]]:
    script_name = normalize_script_name(name)
    title = re.sub(r"\s+", " ", name.strip())
    if len(title) > MAX_SCRIPT_NAME_CHARACTERS:
        raise ValueError(f"Script name must be {MAX_SCRIPT_NAME_CHARACTERS} characters or fewer.")
    normalized_description = description.strip()
    if len(normalized_description) > MAX_SCRIPT_DESCRIPTION_CHARACTERS:
        raise ValueError(f"Script description must be {MAX_SCRIPT_DESCRIPTION_CHARACTERS} characters or fewer.")
    normalized_document = normalize_script_document(document)
    normalized_tags = normalize_script_tags(tags)

    with SCRIPT_LOCK:
        script_dir.mkdir(parents=True, exist_ok=True)
        scripts = load_dialogue_scripts(script_dir)
        existing = scripts.get(script_name)
        if existing is not None and not overwrite:
            raise ValueError(f"Dialogue script '{script_name}' already exists.")

        filename = f"script-{script_name}-{secrets.token_hex(6)}.ssml"
        destination = script_dir / filename
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=script_dir,
                prefix=f".{script_name}.",
                suffix=".tmp",
            ) as output:
                temporary_path = Path(output.name)
                output.write(normalized_document)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None

            now = datetime.now(UTC).isoformat()
            record = {
                "title": title,
                "description": normalized_description,
                "tags": normalized_tags,
                "document_file": filename,
                "turns": max(0, int(turns)),
                "words": max(0, int(words)),
                "created_at": existing.get("created_at", now) if existing else now,
                "updated_at": now,
            }
            scripts[script_name] = record
            _write_index(script_dir, scripts)
        except Exception:
            destination.unlink(missing_ok=True)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        previous = str((existing or {}).get("document_file") or "")
        if previous and previous != filename:
            (script_dir / Path(previous).name).unlink(missing_ok=True)
        return script_name, record


def resolve_dialogue_script(script_dir: Path, name: str) -> tuple[str, dict[str, Any], Path]:
    script_name = normalize_script_name(name)
    with SCRIPT_LOCK:
        record = load_dialogue_scripts(script_dir).get(script_name)
        path = _document_path(script_dir, str((record or {}).get("document_file") or ""))
    if record is None or path is None:
        raise ValueError(f"Dialogue script '{script_name}' does not exist.")
    return script_name, record, path


def delete_dialogue_script(script_dir: Path, name: str) -> str:
    script_name = normalize_script_name(name)
    with SCRIPT_LOCK:
        scripts = load_dialogue_scripts(script_dir)
        record = scripts.pop(script_name, None)
        if record is None:
            raise ValueError(f"Dialogue script '{script_name}' does not exist.")
        _write_index(script_dir, scripts)
        (script_dir / Path(str(record.get("document_file") or "")).name).unlink(missing_ok=True)
    return script_name
