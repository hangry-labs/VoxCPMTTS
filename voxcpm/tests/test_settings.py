from __future__ import annotations

import json

from voxcpm.settings import RuntimeSettingsStore


def test_mcp_settings_are_persisted_together(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = RuntimeSettingsStore(path)
    store.set_mcp_access(enabled=True, advanced_enabled=False)

    assert store.mcp_enabled()
    assert not store.mcp_advanced_enabled()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "mcp_advanced_enabled": False,
        "mcp_enabled": True,
    }
