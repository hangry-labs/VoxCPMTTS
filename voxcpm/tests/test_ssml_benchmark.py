from __future__ import annotations

from benchmarks.ssml.run_benchmark import (
    DIALOGUE_TURNS,
    EXPECTED_TEXT,
    SSML_H_DIALOGUE,
    STANDARD_SSML,
    evaluate_transcript,
    seeds_for_rounds,
)
from voxcpm import app as runtime
from voxcpm.ssml import compile_ssml


def _speech_text(document: str, input_type: str) -> list[str]:
    plan = compile_ssml(
        document,
        input_type,
        default_language="English",
        resolve_language=runtime.resolve_ssml_language,
    )
    return [unit.text for unit in plan.units if unit.kind == "speech"]


def test_paired_documents_compile_to_the_same_ten_turns() -> None:
    expected = list(DIALOGUE_TURNS)

    assert _speech_text(STANDARD_SSML, "ssml") == expected
    assert _speech_text(SSML_H_DIALOGUE, "ssml-h") == expected


def test_complete_transcript_passes_all_dialogue_checks() -> None:
    result = evaluate_transcript(EXPECTED_TEXT)

    assert result["passed"] is True
    assert result["exact"] is True
    assert result["turn_endings_detected"] == 10
    assert result["ordered_turn_endings"] == 10
    assert result["tail_complete"] is True


def test_progressively_truncated_dialogue_fails_with_missing_endings() -> None:
    transcript = " ".join(
        (
            DIALOGUE_TURNS[0],
            "We are testing a complete multi-speaker discussion.",
            DIALOGUE_TURNS[2],
            DIALOGUE_TURNS[3],
            "Let's push it further. Can you introduce a third perspective into",
            "Absolutely. Imagine a moderator stepping in to correct a factual",
            "Nice. Now add some emotion. How would the guest react if the host interrupted",
            DIALOGUE_TURNS[7],
            DIALOGUE_TURNS[8],
            "Agreed. We've successfully demonstrated dynamic voice switching.",
        )
    )

    result = evaluate_transcript(transcript)

    assert result["passed"] is False
    assert result["tail_complete"] is False
    assert result["turn_endings_detected"] == 5
    assert result["missing_turn_endings"] == [
        "from one document",
        "into this dialogue",
        "error mid-sentence",
        "interrupted them",
        "within a single structure",
    ]


def test_fixed_seeds_repeat_for_longer_runs() -> None:
    seeds = seeds_for_rounds(7)

    assert len(seeds) == 7
    assert seeds[0] == seeds[5]
