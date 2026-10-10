from __future__ import annotations

import re


DEFAULT_LONG_TEXT_TARGET = 50
DEFAULT_LONG_TEXT_LIMIT = 80

_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]")
_TEXT_UNIT = re.compile(
    r"[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]|"
    r"[^\W_]+(?:['’][^\W_]+)*",
    re.UNICODE,
)
_SENTENCE_END = set(".!?。！？；;")
_CLAUSE_END = set(",，、:：")
_CLOSING_PUNCTUATION = set("\"'”’）)]}」』》〉")


def count_text_units(text: str) -> int:
    """Count words in spaced scripts and characters in CJK scripts."""

    return sum(1 for _ in _TEXT_UNIT.finditer(text))


def _unit_end_positions(text: str) -> list[int]:
    return [match.end() for match in _TEXT_UNIT.finditer(text)]


def _boundary_after(text: str, index: int) -> int:
    boundary = index + 1
    while boundary < len(text) and text[boundary] in _CLOSING_PUNCTUATION:
        boundary += 1
    return boundary


def _preferred_boundary(text: str, target_position: int, limit_position: int) -> int | None:
    sentence_boundaries = [
        _boundary_after(text, index)
        for index, character in enumerate(text[:limit_position])
        if character in _SENTENCE_END
    ]
    after_target = [position for position in sentence_boundaries if position >= target_position]
    if after_target:
        return after_target[0]

    clause_boundaries = [
        _boundary_after(text, index)
        for index, character in enumerate(text[:limit_position])
        if character in _CLAUSE_END and index + 1 >= target_position
    ]
    if clause_boundaries:
        return clause_boundaries[0]

    whitespace = [
        index
        for index, character in enumerate(text[:limit_position])
        if character.isspace() and index >= target_position
    ]
    return whitespace[0] if whitespace else None


def split_long_text(
    text: str,
    *,
    target_units: int = DEFAULT_LONG_TEXT_TARGET,
    limit_units: int = DEFAULT_LONG_TEXT_LIMIT,
) -> list[str]:
    """Split long synthesis input near complete sentence or clause boundaries."""

    if target_units < 1 or limit_units < target_units:
        raise ValueError("Long-text limits must satisfy 1 <= target_units <= limit_units")

    remaining = text.strip()
    if not remaining:
        return []

    sections: list[str] = []
    while count_text_units(remaining) > target_units:
        unit_ends = _unit_end_positions(remaining)
        target_position = unit_ends[target_units - 1]
        limit_position = unit_ends[min(limit_units, len(unit_ends)) - 1]
        boundary = _preferred_boundary(remaining, target_position, limit_position)
        if boundary is None:
            if len(unit_ends) <= limit_units:
                break
            earlier_sentence_boundaries = [
                _boundary_after(remaining, index)
                for index, character in enumerate(remaining[:limit_position])
                if character in _SENTENCE_END
            ]
            boundary = earlier_sentence_boundaries[-1] if earlier_sentence_boundaries else limit_position

        if not remaining[boundary:].strip():
            break

        section = remaining[:boundary].strip()
        if not section:
            boundary = limit_position
            section = remaining[:boundary].strip()
        sections.append(section)
        remaining = remaining[boundary:].strip()

    if remaining:
        sections.append(remaining)
    return sections


def uses_cjk_units(text: str) -> bool:
    """Return whether the planner counts any script characters individually."""

    return bool(_CJK_CHARACTER.search(text))
