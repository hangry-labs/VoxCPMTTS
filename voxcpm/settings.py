"""Persistent operator settings for the Docker deployment."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS_PATH = "/app/persistent/app/settings.json"
MCP_ENABLED_KEY = "mcp_enabled"
MCP_ADVANCED_ENABLED_KEY = "mcp_advanced_enabled"


class RuntimeSettingsStore:
    """Atomically store the small set of operator-controlled preferences."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or os.getenv("VOXCPM_SETTINGS_PATH")
        self.path = Path(configured_path or DEFAULT_SETTINGS_PATH)
        self._lock = RLock()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable runtime settings %s: %s", self.path, exc)
            return {}
        if not isinstance(payload, dict):
            logger.warning("Ignoring runtime settings %s: expected a JSON object", self.path)
            return {}
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._read_unlocked())

    def get(self, key: str, default: Any = None) -> Any:
        return self.snapshot().get(key, default)

    def set_many(self, values: dict[str, Any]) -> None:
        if not values or any(not key or not isinstance(key, str) for key in values):
            raise ValueError("Runtime setting keys must be non-empty strings.")
        with self._lock:
            payload = self._read_unlocked()
            payload.update(values)
            self._write_unlocked(payload)

    def mcp_enabled(self, *, default: bool = False) -> bool:
        return bool(self.get(MCP_ENABLED_KEY, default))

    def mcp_advanced_enabled(self, *, default: bool = False) -> bool:
        return bool(self.get(MCP_ADVANCED_ENABLED_KEY, default))

    def set_mcp_access(self, *, enabled: bool, advanced_enabled: bool) -> None:
        self.set_many(
            {
                MCP_ENABLED_KEY: bool(enabled),
                MCP_ADVANCED_ENABLED_KEY: bool(advanced_enabled),
            }
        )

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
