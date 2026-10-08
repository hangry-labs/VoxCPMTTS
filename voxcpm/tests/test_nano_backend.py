from __future__ import annotations

import threading

import numpy as np
import pytest

from voxcpm.nano_backend import NanoVoxCPM, _bounded_generation_length


class FakeServer:
    def __init__(self, chunk_counts: dict[int, int]) -> None:
        self.chunk_counts = chunk_counts
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        for _ in range(self.chunk_counts[kwargs["seed"]]):
            yield np.ones(4, dtype=np.float32)


def fake_nano(chunk_counts: dict[int, int]) -> NanoVoxCPM:
    model = object.__new__(NanoVoxCPM)
    model._generation_lock = threading.Lock()
    model._server = FakeServer(chunk_counts)
    model._text_tokenizer = lambda _text: [1, 2]
    model._patch_size = 2
    model.inference_timesteps = 10
    model.last_successful_seed = None
    model.text_normalizer = None
    model.denoiser = None
    return model


def test_generation_length_matches_native_badcase_guard() -> None:
    limit, threshold = _bounded_generation_length("hello", lambda _text: range(9), 2000, 6.0, 2)

    assert limit == 32
    assert threshold == 54.0


def test_non_streaming_generation_retries_overlong_attempt() -> None:
    model = fake_nano({42: 6, 43: 2})

    audio = model.generate(text="hello", seed=42, retry_badcase_max_times=3)

    assert audio.shape == (8,)
    assert model.last_successful_seed == 43
    assert [call["seed"] for call in model._server.calls] == [42, 43]
    assert all(call["max_generate_length"] == 11 for call in model._server.calls)


def test_streaming_generation_is_bounded_without_retry() -> None:
    model = fake_nano({42: 6})

    with pytest.warns(UserWarning, match="not supported in streaming mode"):
        chunks = list(model.generate_streaming(text="hello", seed=42, retry_badcase=True))

    assert len(chunks) == 6
    assert model.last_successful_seed == 42
    assert len(model._server.calls) == 1
