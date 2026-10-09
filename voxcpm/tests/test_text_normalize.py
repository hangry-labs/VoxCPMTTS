# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEXT_NORMALIZE_PATH = ROOT / "voxcpm" / "utils" / "text_normalize.py"


@pytest.fixture
def text_normalize(monkeypatch):
    monkeypatch.setitem(sys.modules, "regex", types.ModuleType("regex"))

    inflect_stub = types.ModuleType("inflect")
    inflect_stub.engine = lambda: None
    monkeypatch.setitem(sys.modules, "inflect", inflect_stub)

    wetext_stub = types.ModuleType("wetext")
    wetext_stub.Normalizer = object
    monkeypatch.setitem(sys.modules, "wetext", wetext_stub)

    spec = importlib.util.spec_from_file_location("voxcpm.utils.text_normalize_test", TEXT_NORMALIZE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ""),
        (" hello", "hello"),
        ("hello ", "hello"),
        (" hello ", "hello"),
        ("hello world", "hello world"),
        ("你 好", "你好"),
        ("你 hello 世界", "你hello世界"),
    ],
)
def test_replace_blank_handles_boundaries_and_preserves_interior_rules(text_normalize, source, expected):
    assert text_normalize.replace_blank(source) == expected
