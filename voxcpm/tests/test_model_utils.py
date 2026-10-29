# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = ROOT / "voxcpm" / "model" / "utils.py"

transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

spec = importlib.util.spec_from_file_location("voxcpm.model.utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(utils)


def test_resolve_runtime_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_mps", lambda: False)

    assert utils.resolve_runtime_device(None, "cuda") == "cpu"


def test_resolve_runtime_device_auto_uses_mps_when_available(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)

    assert utils.resolve_runtime_device("auto", "cuda") == "mps"


def test_resolve_runtime_device_respects_explicit_cpu(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)

    assert utils.resolve_runtime_device("cpu", "cuda") == "cpu"


def test_resolve_runtime_device_rejects_unavailable_explicit_cuda(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)

    with pytest.raises(ValueError, match="CUDA is not available"):
        utils.resolve_runtime_device("cuda:0", "cuda")


def test_materialize_generation_seed_preserves_valid_seed():
    assert utils.materialize_generation_seed(0) == 0
    assert utils.materialize_generation_seed(2**32 - 1) == 2**32 - 1


def test_materialize_generation_seed_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="seed must be between"):
        utils.materialize_generation_seed(2**32)


@pytest.mark.parametrize("piece", ["\u2581公司简介", "\u2581你好", "你好", "hello"])
def test_chinese_tokenizer_normalizes_vocab_and_emitted_pieces_together(piece):
    vocab = {"\u2581公司简介": 100, "\u2581你好": 101, "你好": 102, "hello": 103}
    vocab.update({char: index for index, char in enumerate("公司简介你好", start=1)})

    class Tokenizer:
        def __init__(self):
            self.vocab = vocab

        def tokenize(self, text, **kwargs):
            return [piece]

        def convert_tokens_to_ids(self, tokens):
            return [self.vocab[token] for token in tokens]

    wrapped = utils.mask_multichar_chinese_tokens(Tokenizer())
    expected = ["hello"] if piece == "hello" else list(piece.replace("\u2581", ""))
    assert wrapped.tokenize("input") == expected
    assert wrapped("input") == [vocab[token] for token in expected]
