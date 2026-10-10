from unittest.mock import patch

import numpy as np

import voxcpm.app as runtime
from voxcpm.text_planning import count_text_units, split_long_text, uses_cjk_units


def test_text_units_count_words_and_cjk_characters() -> None:
    assert count_text_units("A calm narrator speaks clearly.") == 5
    assert count_text_units("今天测试中文。") == 6
    assert uses_cjk_units("今天测试中文。") is True
    assert uses_cjk_units("English only.") is False


def test_long_english_prefers_the_first_boundary_after_fifty_words() -> None:
    first = " ".join(f"word{index}" for index in range(55)) + "."
    second = " ".join(f"tail{index}" for index in range(20)) + "."

    sections = split_long_text(f"{first} {second}")

    assert sections == [first, second]
    assert "".join(sections).replace(" ", "") == f"{first}{second}".replace(" ", "")


def test_long_chinese_uses_character_units_and_keeps_sentence_punctuation() -> None:
    first = "中" * 55 + "。"
    second = "文" * 30 + "。"

    sections = split_long_text(first + second)

    assert sections == [first, second]


def test_short_text_is_not_split() -> None:
    text = "This request is already short enough."
    assert split_long_text(text) == [text]


class _FakeTTSModel:
    sample_rate = 100


class _FakeModel:
    tts_model = _FakeTTSModel()

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.full(10, len(self.calls), dtype=np.float32)


def test_generation_protection_reuses_seed_and_inserts_a_short_join() -> None:
    model = _FakeModel()
    first = " ".join(f"word{index}" for index in range(55)) + "."
    second = " ".join(f"tail{index}" for index in range(20)) + "."
    payload = runtime.TTSRequest(text=f"{first} {second}", seed=399, randomize_seed=False)

    with patch.object(runtime, "set_generation_activity"):
        waveform = runtime.generate_planned_waveform(payload, model, 399)

    assert [call["text"] for call in model.calls] == [first, second]
    assert [call["seed"] for call in model.calls] == [399, 399]
    assert waveform.size == 10 + round(100 * runtime.LONG_TEXT_JOIN_SECONDS) + 10


def test_generation_protection_can_be_disabled() -> None:
    model = _FakeModel()
    text = " ".join(f"word{index}" for index in range(100)) + "."
    payload = runtime.TTSRequest(text=text, protect_long_audio=False, seed=12, randomize_seed=False)

    with patch.object(runtime, "set_generation_activity"):
        runtime.generate_planned_waveform(payload, model, 12)

    assert len(model.calls) == 1
    assert model.calls[0]["text"] == text
