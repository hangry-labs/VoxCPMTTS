from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ssml_h import (
    MAX_BREAK_MS,
    MAX_PHONEME_CHARACTERS,
    MAX_SSML_ELEMENTS,
    MAX_SSML_NESTING,
    MAX_SSML_SOURCE_CHARACTERS,
    MAX_SSML_UNITS,
    MAX_SSML_VOICE_DEFINITIONS,
    MAX_TOTAL_BREAK_MS,
    MAX_VOICE_DESCRIPTION_CHARACTERS,
    MAX_VOICE_SAMPLE_CHARACTERS,
    SSML_H_NAMESPACE,
    SSML_NAMESPACE,
    SSMLPlan,
    SSMLProsody,
    SSMLUnit,
    SSMLValidationError,
    SSMLVoiceDefinition,
    compile_ssml as compile_ssml_document,
    ssml_capabilities as base_ssml_capabilities,
)


LanguageResolver = Callable[[str], str]
VoiceValidator = Callable[[str, frozenset[str]], None]
DEFAULT_DYNAMIC_VOICE_SAMPLE_LANGUAGE = "en-US"
DEFAULT_DYNAMIC_VOICE_SAMPLE_TEXT = (
    "Hello, this is my natural speaking voice, kept clear, steady, and consistent "
    "for every conversation we share."
)


def compile_ssml(
    document: str,
    input_type: Literal["ssml", "ssml-h"],
    *,
    default_language: str | None = None,
    default_voice: str | None = None,
    resolve_language: LanguageResolver | None = None,
    validate_voice: VoiceValidator | None = None,
) -> SSMLPlan:
    """Compile a complete SSML document with the shared hardened parser."""

    return compile_ssml_document(
        document,
        input_type,
        default_language=default_language,
        default_voice=default_voice,
        resolve_language=resolve_language,
        validate_voice=validate_voice,
    )


def ssml_capabilities() -> dict:
    """Return the shared profile with VoxCPM-specific feature support."""

    capabilities = base_ssml_capabilities(
        phoneme_alphabets=(),
        description_supported=True,
    )
    capabilities["ssml_h"]["default_voice_sample"] = {
        "language": DEFAULT_DYNAMIC_VOICE_SAMPLE_LANGUAGE,
        "text": DEFAULT_DYNAMIC_VOICE_SAMPLE_TEXT,
        "used_when": "h:sample is omitted",
    }
    capabilities["processor"] = {
        "voice_selection": "Saved voice profile names are accepted by <voice name>.",
        "prosody": "Rate, pitch, and volume are applied to each generated unit.",
        "remote_audio": False,
    }
    return capabilities


__all__ = [
    "MAX_BREAK_MS",
    "MAX_PHONEME_CHARACTERS",
    "MAX_SSML_ELEMENTS",
    "MAX_SSML_NESTING",
    "MAX_SSML_SOURCE_CHARACTERS",
    "MAX_SSML_UNITS",
    "MAX_SSML_VOICE_DEFINITIONS",
    "MAX_TOTAL_BREAK_MS",
    "MAX_VOICE_DESCRIPTION_CHARACTERS",
    "MAX_VOICE_SAMPLE_CHARACTERS",
    "DEFAULT_DYNAMIC_VOICE_SAMPLE_LANGUAGE",
    "DEFAULT_DYNAMIC_VOICE_SAMPLE_TEXT",
    "SSML_H_NAMESPACE",
    "SSML_NAMESPACE",
    "SSMLPlan",
    "SSMLProsody",
    "SSMLUnit",
    "SSMLValidationError",
    "SSMLVoiceDefinition",
    "compile_ssml",
    "ssml_capabilities",
]
