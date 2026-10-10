from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from voxcpm.ssml import (
    DEFAULT_DYNAMIC_VOICE_SAMPLE_LANGUAGE,
    DEFAULT_DYNAMIC_VOICE_SAMPLE_TEXT,
    SSMLPlan,
    SSMLUnit,
    SSMLValidationError,
    SSMLVoiceDefinition,
)


IMPLICIT_HANDOFF_MS = 120
SILENCE_THRESHOLD = 192
EDGE_SCAN_MS = 600
EDGE_KEEP_MS = 35


@dataclass(frozen=True)
class SSMLVoiceBinding:
    name: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    control: str | None = None
    language: str | None = None
    generation_seed: int | None = None


@dataclass(frozen=True)
class PreparedSSMLVoice:
    definition: SSMLVoiceDefinition
    binding: SSMLVoiceBinding
    sample_text: str
    sample_language: str | None
    seed: int


GenerateSpeech = Callable[[SSMLUnit, SSMLVoiceBinding, int], tuple[int, np.ndarray]]
PrepareVoice = Callable[[SSMLVoiceDefinition, str, str | None, int, Path], SSMLVoiceBinding]
ResolveVoice = Callable[[str], SSMLVoiceBinding]
ResolveLanguage = Callable[[str], str]
CommitProfiles = Callable[[list[PreparedSSMLVoice], Path], dict[str, str]]
AudioTimelineItem = tuple[Literal["speech", "break"], np.ndarray | int]


def _to_int16(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio).reshape(-1)
    if array.dtype == np.int16:
        return array
    if np.issubdtype(array.dtype, np.integer):
        maximum = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / maximum
    return (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)


def _atempo_filters(multiplier: float) -> list[str]:
    filters: list[str] = []
    current = multiplier
    while current > 2.0:
        filters.append("atempo=2.0")
        current /= 2.0
    while current < 0.5:
        filters.append("atempo=0.5")
        current /= 0.5
    filters.append(f"atempo={current:.6f}")
    return filters


def apply_audio_effects(
    audio: np.ndarray,
    sample_rate: int,
    *,
    rate: float,
    pitch_semitones: float,
    volume: float,
) -> np.ndarray:
    filters: list[str] = []
    if abs(pitch_semitones) > 0.001:
        pitch_factor = 2 ** (pitch_semitones / 12)
        shifted_rate = max(1, round(sample_rate * pitch_factor))
        filters.extend((f"asetrate={shifted_rate}", f"aresample={sample_rate}"))
        filters.extend(_atempo_filters(1 / pitch_factor))
    if abs(rate - 1.0) > 0.001:
        filters.extend(_atempo_filters(rate))
    if abs(volume - 1.0) > 0.001:
        filters.append(f"volume={volume:.6f}")
    waveform = _to_int16(audio)
    if not filters or waveform.size == 0:
        return waveform

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
        "-i", "pipe:0", "-af", ",".join(filters),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=waveform.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for SSML prosody controls") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to apply SSML prosody: {detail}") from exc
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.int16, copy=True)


def _trim_edge(audio: np.ndarray, sample_rate: int, *, leading: bool) -> np.ndarray:
    waveform = _to_int16(audio)
    if waveform.size == 0:
        return waveform
    scan = min(waveform.size, round(sample_rate * EDGE_SCAN_MS / 1000))
    keep = min(scan, round(sample_rate * EDGE_KEEP_MS / 1000))
    section = waveform[:scan] if leading else waveform[-scan:]
    active = np.flatnonzero(np.abs(section.astype(np.int32)) > SILENCE_THRESHOLD)
    if active.size == 0:
        return waveform
    if leading:
        return waveform[max(0, int(active[0]) - keep):]
    trailing_silence = scan - int(active[-1]) - 1
    trim = max(0, trailing_silence - keep)
    return waveform if trim == 0 else waveform[:-trim]


def _fade(audio: np.ndarray, sample_rate: int, *, leading: bool, trailing: bool) -> np.ndarray:
    waveform = _to_int16(audio).copy()
    fade_samples = min(round(sample_rate * 0.01), waveform.size)
    if leading and fade_samples:
        waveform[:fade_samples] = (
            waveform[:fade_samples].astype(np.float32)
            * np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        ).astype(np.int16)
    if trailing and fade_samples:
        waveform[-fade_samples:] = (
            waveform[-fade_samples:].astype(np.float32)
            * np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        ).astype(np.int16)
    return waveform


def iter_assembled_audio(items: Iterator[AudioTimelineItem], sample_rate: int) -> Iterator[np.ndarray]:
    """Assemble semantic speech takes and breaks using the normal SSML boundary policy."""
    pending_audio: np.ndarray | None = None
    pending_break_ms: int | None = None
    leading_break_ms = 0
    for kind, value in items:
        if kind == "break":
            duration_ms = int(value)
            if pending_audio is None:
                leading_break_ms += duration_ms
            else:
                pending_break_ms = (pending_break_ms or 0) + duration_ms
            continue

        current = _to_int16(np.asarray(value))
        if pending_audio is None:
            if leading_break_ms:
                yield np.zeros(round(sample_rate * leading_break_ms / 1000), dtype=np.int16)
                leading_break_ms = 0
            pending_audio = current
            continue

        yield _trim_edge(pending_audio, sample_rate, leading=False)
        current = _trim_edge(current, sample_rate, leading=True)
        gap_ms = pending_break_ms if pending_break_ms is not None else IMPLICIT_HANDOFF_MS
        if gap_ms:
            yield np.zeros(round(sample_rate * gap_ms / 1000), dtype=np.int16)
        pending_audio = current
        pending_break_ms = None

    if pending_audio is not None:
        yield pending_audio
        if pending_break_ms:
            yield np.zeros(round(sample_rate * pending_break_ms / 1000), dtype=np.int16)
    elif leading_break_ms:
        yield np.zeros(round(sample_rate * leading_break_ms / 1000), dtype=np.int16)


def assemble_audio_timeline(items: Iterator[AudioTimelineItem], sample_rate: int) -> np.ndarray:
    chunks = list(iter_assembled_audio(items, sample_rate))
    waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    return _fade(waveform, sample_rate, leading=True, trailing=True)


class SSMLExecutionSession:
    """Execute one compiled SSML document and own its temporary voice assets."""

    def __init__(
        self,
        *,
        plan: SSMLPlan,
        default_binding: SSMLVoiceBinding,
        request_seed: int,
        request_speed: float,
        default_sample_rate: int,
        staging_parent: Path,
        resolve_voice: ResolveVoice,
        resolve_language: ResolveLanguage,
        prepare_voice: PrepareVoice,
        generate_speech: GenerateSpeech,
        commit_profiles: CommitProfiles,
    ) -> None:
        self.plan = plan
        self.default_binding = default_binding
        self.request_seed = request_seed
        self.request_speed = request_speed
        self.default_sample_rate = default_sample_rate
        self.resolve_voice = resolve_voice
        self.resolve_language = resolve_language
        self.prepare_voice = prepare_voice
        self.generate_speech = generate_speech
        self.commit_profiles_callback = commit_profiles
        staging_parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir = Path(tempfile.mkdtemp(prefix="ssml-h-", dir=staging_parent))
        self.dynamic_voices: dict[str, PreparedSSMLVoice] = {}
        self.resolved_voices: dict[str, SSMLVoiceBinding] = {}
        self.sample_rate: int | None = None
        self.committed_profiles: dict[str, str] = {}
        self._committed = False
        self._closed = False

    def prepare(self) -> None:
        for index, definition in enumerate(self.plan.voice_definitions):
            sample_text, sample_language = self._bootstrap_sample(definition)
            seed = definition.seed if definition.seed is not None else (self.request_seed + index) % (2**32)
            binding = self.prepare_voice(definition, sample_text, sample_language, seed, self.staging_dir)
            self.dynamic_voices[definition.name] = PreparedSSMLVoice(
                definition=definition,
                binding=binding,
                sample_text=sample_text,
                sample_language=sample_language,
                seed=seed,
            )

    def _bootstrap_sample(self, definition: SSMLVoiceDefinition) -> tuple[str, str | None]:
        if definition.sample:
            language = self.resolve_language(definition.sample_language) if definition.sample_language else None
            if language is None and definition.languages:
                language = self.resolve_language(definition.languages[0])
            return definition.sample, language
        return DEFAULT_DYNAMIC_VOICE_SAMPLE_TEXT, self.resolve_language(DEFAULT_DYNAMIC_VOICE_SAMPLE_LANGUAGE)

    def _binding_for_unit(self, unit: SSMLUnit) -> SSMLVoiceBinding:
        if not unit.voice:
            return self.default_binding
        dynamic = self.dynamic_voices.get(unit.voice)
        if dynamic is not None:
            return dynamic.binding
        if unit.voice not in self.resolved_voices:
            self.resolved_voices[unit.voice] = self.resolve_voice(unit.voice)
        return self.resolved_voices[unit.voice]

    def _render_speech(self, unit: SSMLUnit, index: int) -> np.ndarray:
        rate = self.request_speed * unit.prosody.rate
        pitch = unit.prosody.pitch_semitones
        volume = unit.prosody.volume
        if not 0.25 <= rate <= 4.0:
            raise SSMLValidationError(f"Effective rate for SSML unit {index + 1} must be between 0.25 and 4.0.")
        if not -12.0 <= pitch <= 12.0:
            raise SSMLValidationError(f"Effective pitch for SSML unit {index + 1} must be between -12st and +12st.")
        if not 0.0 <= volume <= 2.0:
            raise SSMLValidationError(f"Effective volume for SSML unit {index + 1} must be between 0 and 2.0.")
        binding = self._binding_for_unit(unit)
        if unit.direction:
            binding = replace(binding, ref_text=None, control=unit.direction)
        seed = binding.generation_seed
        if seed is None:
            seed = (self.request_seed + len(self.plan.voice_definitions) + index) % (2**32)
        sample_rate, waveform = self.generate_speech(unit, binding, seed)
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif self.sample_rate != sample_rate:
            raise RuntimeError("SSML synthesis units returned inconsistent sample rates.")
        return apply_audio_effects(waveform, sample_rate, rate=rate, pitch_semitones=pitch, volume=volume)

    def speech_seed(self, unit: SSMLUnit, index: int) -> int:
        binding = self._binding_for_unit(unit)
        if binding.generation_seed is not None:
            return binding.generation_seed
        return (self.request_seed + len(self.plan.voice_definitions) + index) % (2**32)

    def render_speech_take(self, speech_index: int) -> tuple[int, np.ndarray, int]:
        if speech_index < 0:
            raise ValueError("Speech take index cannot be negative.")
        current_index = 0
        for unit in self.plan.units:
            if unit.kind != "speech":
                continue
            if current_index == speech_index:
                waveform = self._render_speech(unit, current_index)
                return self.sample_rate or self.default_sample_rate, waveform, self.speech_seed(unit, current_index)
            current_index += 1
        raise ValueError(f"Speech take index {speech_index} is outside this document.")

    def _assembled_chunks(self) -> Iterator[np.ndarray]:
        def timeline() -> Iterator[AudioTimelineItem]:
            speech_index = 0
            for unit in self.plan.units:
                if unit.kind == "break":
                    yield "break", unit.duration_ms
                    continue
                yield "speech", self._render_speech(unit, speech_index)
                speech_index += 1

        yield from iter_assembled_audio(timeline(), self.default_sample_rate)

    def render_array(self) -> tuple[int, np.ndarray]:
        chunks = list(self._assembled_chunks())
        sample_rate = self.sample_rate or self.default_sample_rate
        waveform = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return sample_rate, _fade(waveform, sample_rate, leading=True, trailing=True)

    def iter_chunks(self) -> Iterator[np.ndarray]:
        iterator = iter(self._assembled_chunks())
        try:
            pending = next(iterator)
        except StopIteration:
            return
        first = True
        for current in iterator:
            yield _fade(pending, self.sample_rate or self.default_sample_rate, leading=first, trailing=False)
            first = False
            pending = current
        yield _fade(pending, self.sample_rate or self.default_sample_rate, leading=first, trailing=True)

    def commit_profiles(self) -> dict[str, str]:
        if self._committed:
            return dict(self.committed_profiles)
        persistent = [
            prepared for prepared in self.dynamic_voices.values()
            if prepared.definition.scope == "profile"
        ]
        self.committed_profiles = self.commit_profiles_callback(persistent, self.staging_dir)
        self._committed = True
        return dict(self.committed_profiles)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.dynamic_voices.clear()
        shutil.rmtree(self.staging_dir, ignore_errors=True)


__all__ = [
    "IMPLICIT_HANDOFF_MS",
    "PreparedSSMLVoice",
    "SSMLExecutionSession",
    "SSMLVoiceBinding",
    "apply_audio_effects",
    "assemble_audio_timeline",
    "iter_assembled_audio",
]
