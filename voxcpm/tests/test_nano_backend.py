from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voxcpm.nano_backend import NanoVoxCPM, _bounded_generation_length, _nano_gpu_memory_utilization


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
    model.inference_timesteps = 10
    model.last_successful_seed = None
    model.text_normalizer = None
    model.denoiser = None
    return model


def test_generation_length_uses_nano_steps_with_a_separate_retry_threshold() -> None:
    limit, threshold = _bounded_generation_length("hello", lambda _text: range(9), 2000, 4.5)

    assert limit == 64
    assert threshold == 40.5


@pytest.mark.parametrize(
    ("total_gib", "expected"),
    [
        (8, 0.90),
        (12, 2 / 3),
        (16, 0.50),
        (24, 1 / 3),
    ],
)
def test_auto_gpu_memory_utilization_targets_eight_gib(
    monkeypatch: pytest.MonkeyPatch,
    total_gib: int,
    expected: float,
) -> None:
    monkeypatch.delenv("VOXCPM_NANO_GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=total_gib * 1024**3),
    )

    assert _nano_gpu_memory_utilization(0) == pytest.approx(expected)


def test_explicit_gpu_memory_utilization_overrides_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOXCPM_NANO_GPU_MEMORY_UTILIZATION", "0.61")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: pytest.fail("explicit utilization must not inspect GPU memory"),
    )

    assert _nano_gpu_memory_utilization(0) == 0.61


@pytest.mark.parametrize("value", ["0", "1.01", "invalid"])
def test_invalid_gpu_memory_utilization_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("VOXCPM_NANO_GPU_MEMORY_UTILIZATION", value)

    with pytest.raises(ValueError):
        _nano_gpu_memory_utilization(0)


def test_non_streaming_generation_retries_overlong_attempt() -> None:
    model = fake_nano({42: 10, 43: 2})

    audio = model.generate(text="hello", seed=42, retry_badcase_max_times=3)

    assert audio.shape == (8,)
    assert model.last_successful_seed == 43
    assert [call["seed"] for call in model._server.calls] == [42, 43]
    assert all(call["max_generate_length"] == 22 for call in model._server.calls)


def test_streaming_generation_is_bounded_without_retry() -> None:
    model = fake_nano({42: 10})

    with pytest.warns(UserWarning, match="not supported in streaming mode"):
        chunks = list(model.generate_streaming(text="hello", seed=42, retry_badcase=True))

    assert len(chunks) == 10
    assert model.last_successful_seed == 42
    assert len(model._server.calls) == 1


def test_denoising_reference_audio_does_not_normalize_loudness() -> None:
    model = fake_nano({42: 1})
    denoiser_calls: list[tuple[str, bool]] = []

    class FakeDenoiser:
        def enhance(self, input_path: str, *, output_path: str, normalize_loudness: bool) -> str:
            denoiser_calls.append((input_path, normalize_loudness))
            return output_path

    model.denoiser = FakeDenoiser()
    model._encode_audio = lambda path: path

    model.generate(
        text="hello",
        prompt_wav_path="prompt.wav",
        prompt_text="prompt",
        reference_wav_path="reference.wav",
        denoise=True,
        retry_badcase=False,
        seed=42,
    )

    assert denoiser_calls == [("prompt.wav", False), ("reference.wav", False)]
