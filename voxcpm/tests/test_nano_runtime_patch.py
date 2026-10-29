from __future__ import annotations

import importlib.util
import shutil
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest


@pytest.fixture
def runtime_patch(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "patch-nanovllm-voxcpm.py"
    spec = importlib.util.spec_from_file_location("nano_runtime_patch_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    try:
        package = distribution(module.PACKAGE)
    except PackageNotFoundError:
        pytest.skip("Requires the pinned Nano runtime image")
    if package.version != module.EXPECTED_VERSION:
        pytest.skip("Requires the pinned Nano wheel version")
    root = package.locate_file("nanovllm_voxcpm")
    for relative in (
        "models/voxcpm2/runner.py",
        "models/voxcpm2/utils.py",
        "engine/model_runner.py",
        "engine/block_manager.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    monkeypatch.setattr(module, "package_root", lambda: tmp_path)
    return module, tmp_path


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*.py")}


def test_runtime_patch_updates_real_tokenizer_source_idempotently(runtime_patch):
    patch, root = runtime_patch
    patch.main()
    tokenizer = root / "models" / "voxcpm2" / "utils.py"
    assert patch.PATCHED_CHINESE_TOKEN_LOOKUP in tokenizer.read_text(encoding="utf-8")
    first = snapshot(root)
    patch.main()
    assert snapshot(root) == first


def test_runtime_patch_rejects_unknown_tokenizer_before_writing(runtime_patch):
    patch, root = runtime_patch
    tokenizer = root / "models" / "voxcpm2" / "utils.py"
    source = tokenizer.read_text(encoding="utf-8")
    source = source.replace(patch.ORIGINAL_CHINESE_TOKEN_LOOKUP, "    multichar_tokens = set()\n")
    source = source.replace(patch.PATCHED_CHINESE_TOKEN_LOOKUP, "    multichar_tokens = set()\n")
    tokenizer.write_text(source, encoding="utf-8")
    original = snapshot(root)
    with pytest.raises(RuntimeError, match="Chinese tokenizer lookup"):
        patch.main()
    assert snapshot(root) == original


def test_runtime_patch_rejects_other_wheel_versions_without_writing(runtime_patch, monkeypatch):
    patch, root = runtime_patch
    original = snapshot(root)
    monkeypatch.setattr(patch, "version", lambda package: "9.9.9")
    with pytest.raises(RuntimeError, match="Refusing to patch"):
        patch.main()
    assert snapshot(root) == original
