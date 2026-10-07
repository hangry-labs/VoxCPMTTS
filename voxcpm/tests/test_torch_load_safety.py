"""Prevent unsafe pickle-capable checkpoint loads from re-entering the runtime."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_DIRS = [REPO_ROOT / "voxcpm", REPO_ROOT / "scripts"]


def _python_files():
    for directory in SCANNED_DIRS:
        if directory.is_dir():
            yield from directory.rglob("*.py")


def _is_torch_load(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _has_weights_only_true(node: ast.Call) -> bool:
    return any(
        keyword.arg == "weights_only"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in node.keywords
    )


def test_every_torch_load_sets_weights_only_true():
    offenders = []
    checked = 0
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_torch_load(node):
                checked += 1
                if not _has_weights_only_true(node):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert checked > 0, "expected at least one torch.load call"
    assert not offenders, "torch.load without weights_only=True:\n  " + "\n  ".join(offenders)


def test_torch_load_weights_only_blocks_malicious_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed.txt"

    class Exploit:
        def __reduce__(self):
            import pathlib

            return (pathlib.Path.write_text, (marker, "executed\n"))

    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": {"weight": torch.zeros(1)}, "payload": Exploit()}, checkpoint)

    with pytest.raises(Exception):
        torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert not marker.exists()
