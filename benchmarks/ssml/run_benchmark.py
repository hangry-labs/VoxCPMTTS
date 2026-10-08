from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import time
import unicodedata
import urllib.request
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TTS_URL = os.getenv("VOXCPM_SSML_BENCHMARK_TTS_URL", "http://127.0.0.1:8808")
DEFAULT_ASR_URL = os.getenv("VOXCPM_SSML_BENCHMARK_ASR_URL", "http://127.0.0.1:8000")
DEFAULT_ROUNDS = int(os.getenv("VOXCPM_SSML_BENCHMARK_ROUNDS", "5"))
DEFAULT_RESULTS = ROOT / "benchmarks" / "ssml" / "runs.json"
DEFAULT_SUMMARY = ROOT / "benchmarks" / "ssml" / "BENCHMARKS.md"
DEFAULT_DETAILS = ROOT / "benchmarks" / "ssml" / "DETAILS.md"

DIALOGUE_TURNS = (
    "Welcome to the show. What are we exploring today?",
    "We are testing a complete multi-speaker discussion from one document.",
    "That sounds efficient. Do you think it captures the natural flow of conversation?",
    "I believe so, as long as the context switches are handled cleanly between tags.",
    "Let's push it further. Can you introduce a third perspective into this dialogue?",
    "Absolutely. Imagine a moderator stepping in to correct a factual error mid-sentence.",
    "Nice. Now add some emotion. How would the guest react if the host interrupted them?",
    "They might sigh audibly before responding, adding a layer of realism to the text.",
    "Perfect. Let’s wrap up with a summary statement from the host.",
    "Agreed. We’ve successfully demonstrated dynamic voice switching within a single structure.",
)
EXPECTED_TEXT = " ".join(DIALOGUE_TURNS)
TURN_ENDINGS = (
    "exploring today",
    "from one document",
    "flow of conversation",
    "cleanly between tags",
    "into this dialogue",
    "error mid-sentence",
    "interrupted them",
    "realism to the text",
    "statement from the host",
    "within a single structure",
)
FIXED_SEEDS = (42, 314_159, 2_717_518_076, 20_261_008, 42_424_242)
WARMUP_SEED = 9_001

STANDARD_SSML = """<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  Welcome to the show. What are we exploring today?<break time="120ms"/>
  We are testing a complete multi-speaker discussion from one document.<break time="120ms"/>
  That sounds efficient. Do you think it captures the natural flow of conversation?<break time="120ms"/>
  I believe so, as long as the context switches are handled cleanly between tags.<break time="120ms"/>
  Let's push it further. Can you introduce a third perspective into this dialogue?<break time="120ms"/>
  Absolutely. Imagine a moderator stepping in to correct a factual error mid-sentence.<break time="120ms"/>
  Nice. Now add some emotion. How would the guest react if the host interrupted them?<break time="120ms"/>
  They might sigh audibly before responding, adding a layer of realism to the text.<break time="120ms"/>
  Perfect. Let’s wrap up with a summary statement from the host.<break time="120ms"/>
  Agreed. We’ve successfully demonstrated dynamic voice switching within a single structure.
</speak>"""

SSML_H_DIALOGUE = """<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
  xmlns:h="https://hangrylabs.app/ns/ssml-h/1.0" xml:lang="en-US">
  <metadata><h:extensions version="1.0">
    <h:voice-definition name="Host" gender="female" style="warm and confident">
      <h:sample xml:lang="en-US">Welcome. I will guide our conversation today.</h:sample>
    </h:voice-definition>
    <h:voice-definition name="Guest">
      <h:description>A thoughtful male guest with a relaxed, conversational delivery.</h:description>
    </h:voice-definition>
  </h:extensions></metadata>
  <voice name="Host">Welcome to the show. What are we exploring today?</voice>
  <voice name="Guest">We are testing a complete multi-speaker discussion from one document.</voice>
  <voice name="Host">That sounds efficient. Do you think it captures the natural flow of conversation?</voice>
  <voice name="Guest">I believe so, as long as the context switches are handled cleanly between tags.</voice>
  <voice name="Host">Let's push it further. Can you introduce a third perspective into this dialogue?</voice>
  <voice name="Guest">Absolutely. Imagine a moderator stepping in to correct a factual error mid-sentence.</voice>
  <voice name="Host">Nice. Now add some emotion. How would the guest react if the host interrupted them?</voice>
  <voice name="Guest">They might sigh audibly before responding, adding a layer of realism to the text.</voice>
  <voice name="Host">Perfect. Let’s wrap up with a summary statement from the host.</voice>
  <voice name="Guest">Agreed. We’ve successfully demonstrated dynamic voice switching within a single structure.</voice>

</speak>"""

SCENARIOS = (
    {"id": "ssml", "label": "Standard SSML", "input_type": "ssml", "document": STANDARD_SSML},
    {"id": "ssml-h", "label": "SSML-H", "input_type": "ssml-h", "document": SSML_H_DIALOGUE},
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def request_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(tts_url: str, asr_url: str) -> None:
    deadline = time.monotonic() + 240
    pending = {"TTS": f"{tts_url}/tts/ping", "ASR": f"{asr_url}/health/ready"}
    while pending and time.monotonic() < deadline:
        for name, url in list(pending.items()):
            try:
                request_json(url, timeout=5)
                del pending[name]
            except Exception:
                continue
        if pending:
            time.sleep(2)
    if pending:
        raise RuntimeError(f"Services did not become ready: {', '.join(pending)}")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = []
    current = []
    for character in normalized:
        if unicodedata.category(character)[0] in {"L", "N"}:
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return " ".join(words)


def similarity(expected: str, actual: str) -> float:
    return round(100.0 * difflib.SequenceMatcher(None, normalize_text(expected), normalize_text(actual)).ratio(), 2)


def evaluate_transcript(transcript: str, minimum_similarity: float = 95.0) -> dict[str, Any]:
    actual = normalize_text(transcript)
    expected = normalize_text(EXPECTED_TEXT)
    markers = [normalize_text(marker) for marker in TURN_ENDINGS]
    hits = [marker in actual for marker in markers]
    cursor = 0
    ordered_hits = 0
    for marker in markers:
        position = actual.find(marker, cursor)
        if position >= 0:
            ordered_hits += 1
            cursor = position + len(marker)
    score = similarity(EXPECTED_TEXT, transcript)
    tail_complete = actual.endswith(markers[-1])
    return {
        "exact": actual == expected,
        "similarity_percent": score,
        "turn_endings_detected": sum(hits),
        "ordered_turn_endings": ordered_hits,
        "missing_turn_endings": [TURN_ENDINGS[index] for index, hit in enumerate(hits) if not hit],
        "tail_complete": tail_complete,
        "passed": score >= minimum_similarity and all(hits) and ordered_hits == len(markers) and tail_complete,
    }


def seeds_for_rounds(rounds: int) -> list[int]:
    if rounds < 1:
        raise ValueError("--rounds must be at least 1")
    return [FIXED_SEEDS[index % len(FIXED_SEEDS)] for index in range(rounds)]


def wav_duration(audio: bytes) -> tuple[float, int]:
    with wave.open(io.BytesIO(audio), "rb") as source:
        return source.getnframes() / source.getframerate(), source.getframerate()


def generate_audio(tts_url: str, scenario: dict[str, str], seed: int) -> tuple[bytes, dict[str, str], float]:
    payload = {
        "input_type": scenario["input_type"],
        "text": scenario["document"],
        "language": "English",
        "output_format": "wav",
        "seed": seed,
        "randomize_seed": False,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": False,
        "denoise": False,
    }
    request = urllib.request.Request(
        f"{tts_url}/tts/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1_800) as response:
        audio = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
        content_type = response.headers.get("Content-Type", "")
    elapsed = time.perf_counter() - started
    if not content_type.startswith("audio/wav") or len(audio) < 1_000:
        raise RuntimeError(
            f"Invalid WAV response for {scenario['id']} seed {seed}: "
            f"{content_type!r}, {len(audio)} bytes"
        )
    return audio, headers, elapsed


def multipart_body(audio: bytes, filename: str) -> tuple[bytes, str]:
    boundary = f"----VoxCPMSSMLBenchmark{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in (
        ("model", "qwen3-asr"),
        ("language", "English"),
        ("response_format", "verbose_json"),
        ("temperature", "0"),
    ):
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}.wav"\r\n'.encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            audio,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), boundary


def transcribe_audio(asr_url: str, audio: bytes, filename: str) -> tuple[str, float]:
    body, boundary = multipart_body(audio, filename)
    request = urllib.request.Request(
        f"{asr_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("text", "")).strip(), time.perf_counter() - started


def generate_call(
    tts_url: str,
    scenario: dict[str, str],
    seed: int,
    round_number: int,
    staging_dir: Path,
) -> dict[str, Any]:
    audio, headers, tts_seconds = generate_audio(tts_url, scenario, seed)
    audio_seconds, sample_rate = wav_duration(audio)
    staged_audio = staging_dir / f"{scenario['id']}-r{round_number:02d}-{seed}.wav"
    staged_audio.write_bytes(audio)
    return {
        "scenario": scenario["id"],
        "round": round_number,
        "seed": seed,
        "response_seed": headers.get("x-voxcpm-seed"),
        "tts_seconds": round(tts_seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "rtf": round(tts_seconds / audio_seconds, 4),
        "sample_rate": sample_rate,
        "bytes": len(audio),
        "sha256": hashlib.sha256(audio).hexdigest(),
        "staged_audio": str(staged_audio),
    }


def transcribe_call(asr_url: str, result: dict[str, Any], minimum_similarity: float) -> dict[str, Any]:
    staged_audio = Path(result["staged_audio"])
    transcript, asr_seconds = transcribe_audio(asr_url, staged_audio.read_bytes(), staged_audio.stem)
    result.update({"transcript": transcript, "asr_seconds": round(asr_seconds, 3)})
    result.update(evaluate_transcript(transcript, minimum_similarity))
    result.pop("staged_audio", None)
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    tts_seconds = [float(item["tts_seconds"]) for item in measurements]
    rtf = [float(item["rtf"]) for item in measurements]
    similarities = [float(item["similarity_percent"]) for item in measurements]
    ending_counts = [int(item["turn_endings_detected"]) for item in measurements]
    passed = sum(bool(item["passed"]) for item in measurements)
    return {
        "rounds": len(measurements),
        "passed": passed,
        "pass_percent": round(100.0 * passed / len(measurements), 2),
        "exact_percent": round(100.0 * sum(bool(item["exact"]) for item in measurements) / len(measurements), 2),
        "tail_complete_percent": round(
            100.0 * sum(bool(item["tail_complete"]) for item in measurements) / len(measurements), 2
        ),
        "mean_similarity_percent": round(statistics.mean(similarities), 2),
        "minimum_similarity_percent": round(min(similarities), 2),
        "mean_turn_ending_percent": round(10.0 * statistics.mean(ending_counts), 2),
        "minimum_turn_endings": min(ending_counts),
        "mean_tts_seconds": round(statistics.mean(tts_seconds), 3),
        "p95_tts_seconds": round(percentile(tts_seconds, 0.95), 3),
        "mean_audio_seconds": round(statistics.mean(float(item["audio_seconds"]) for item in measurements), 3),
        "median_rtf": round(statistics.median(rtf), 4),
        "mean_asr_seconds": round(statistics.mean(float(item["asr_seconds"]) for item in measurements), 3),
    }


def append_json(path: Path, run: dict[str, Any]) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": 1, "runs": []}
    data.setdefault("schema", 1)
    data.setdefault("runs", []).append(run)
    data["latest"] = run
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        temporary = Path(file.name)
    temporary.replace(path)


def run_label(run: dict[str, Any]) -> str:
    timestamp = datetime.fromisoformat(run["started_at"]).astimezone()
    return f"{timestamp:%d.%m.%Y %H:%M:%S} - {run['tts_version']}"


def append_summary(path: Path, run: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8").rstrip("\n")
    rows = []
    for scenario in SCENARIOS:
        summary = run["summary"][scenario["id"]]
        rows.append(
            f"| {run_label(run)} | {scenario['label']} | {summary['rounds']} | {summary['passed']} | "
            f"{summary['pass_percent']:.2f}% | {summary['tail_complete_percent']:.2f}% | "
            f"{summary['mean_turn_ending_percent']:.2f}% | {summary['mean_similarity_percent']:.2f}% | "
            f"{summary['minimum_similarity_percent']:.2f}% | {summary['mean_tts_seconds']:.3f} | "
            f"{summary['median_rtf']:.4f} | {run['asr_model'].replace('|', '/')} |"
        )
    path.write_text(existing + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def append_details(path: Path, run: dict[str, Any]) -> None:
    lines = [
        "",
        f"## {run_label(run)}",
        "",
        f"- TTS runtime: `{run['hardware']}`",
        f"- TTS build: `{run['tts_build_id']}`",
        f"- ASR model: `{run['asr_model']}`",
        f"- Minimum transcript similarity: `{run['minimum_similarity_percent']:.2f}%`",
        f"- Comment: {run.get('comment') or 'None'}",
        "",
        "| Mode | Round | Seed | Result | Tail | Endings | Similarity | TTS seconds | "
        "Audio seconds | RTF | ASR seconds | Transcript | SHA-256 |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for measurement in run["measurements"]:
        transcript = str(measurement["transcript"]).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {measurement['scenario']} | {measurement['round']} | {measurement['seed']} | "
            f"{'pass' if measurement['passed'] else 'FAIL'} | {'yes' if measurement['tail_complete'] else 'NO'} | "
            f"{measurement['turn_endings_detected']}/10 | {measurement['similarity_percent']:.2f}% | "
            f"{measurement['tts_seconds']:.3f} | {measurement['audio_seconds']:.3f} | "
            f"{measurement['rtf']:.4f} | {measurement['asr_seconds']:.3f} | {transcript} | "
            f"`{measurement['sha256']}` |"
        )
        if measurement["missing_turn_endings"]:
            missing = ", ".join(measurement["missing_turn_endings"])
            lines.append(f"|  |  |  | Missing endings |  |  |  |  |  |  |  | {missing} |  |")
    existing = path.read_text(encoding="utf-8").rstrip("\n")
    path.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark paired ten-turn SSML and SSML-H dialogue generation.")
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL)
    parser.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--minimum-similarity", type=float, default=95.0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--comment", default="")
    parser.add_argument("--no-write", action="store_true", help="Run without changing committed benchmark history.")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not 0 <= args.minimum_similarity <= 100:
        parser.error("--minimum-similarity must be between 0 and 100")
    return args


def main() -> int:
    args = parse_args()
    wait_ready(args.tts_url, args.asr_url)
    tts_status = request_json(f"{args.tts_url}/tts/status")
    asr_models = request_json(f"{args.asr_url}/v1/models").get("data", [])
    asr_model = str(asr_models[0].get("id")) if asr_models else "unknown"
    seeds = seeds_for_rounds(args.rounds)
    started = time.perf_counter()
    started_at = now_iso()

    with tempfile.TemporaryDirectory(prefix="voxcpmtts-ssml-") as staging_root:
        staging_dir = Path(staging_root)
        warmups: list[dict[str, Any]] = []
        measurements: list[dict[str, Any]] = []
        print("Phase 1/2: generating warmup and measured WAV files.", flush=True)
        for scenario in SCENARIOS:
            warmup = generate_call(args.tts_url, scenario, WARMUP_SEED, 0, staging_dir)
            warmups.append(warmup)
            print(
                f"[TTS warmup {scenario['id']}] {warmup['tts_seconds']:.3f}s, "
                f"{warmup['audio_seconds']:.3f}s audio",
                flush=True,
            )
            for round_number, seed in enumerate(seeds, start=1):
                result = generate_call(args.tts_url, scenario, seed, round_number, staging_dir)
                measurements.append(result)
                print(
                    f"[TTS {scenario['id']} {round_number:02d}/{args.rounds:02d}] seed={seed} "
                    f"tts={result['tts_seconds']:.3f}s rtf={result['rtf']:.4f}",
                    flush=True,
                )

        print("Phase 2/2: warming Qwen3-ASR, then judging every measured WAV.", flush=True)
        for warmup in warmups:
            transcribe_call(args.asr_url, warmup, args.minimum_similarity)
            print(
                f"[ASR warmup {warmup['scenario']}] endings={warmup['turn_endings_detected']}/10 "
                f"tail={'yes' if warmup['tail_complete'] else 'no'}",
                flush=True,
            )
        for index, result in enumerate(measurements, start=1):
            transcribe_call(args.asr_url, result, args.minimum_similarity)
            print(
                f"[ASR {index:02d}/{len(measurements):02d}] {result['scenario']} "
                f"{'PASS' if result['passed'] else 'FAIL'} endings={result['turn_endings_detected']}/10 "
                f"tail={'yes' if result['tail_complete'] else 'no'} similarity={result['similarity_percent']:.2f}%",
                flush=True,
            )

    summary = {
        scenario["id"]: summarize([item for item in measurements if item["scenario"] == scenario["id"]])
        for scenario in SCENARIOS
    }
    run = {
        "started_at": started_at,
        "total_seconds": round(time.perf_counter() - started, 3),
        "tts_version": str(tts_status.get("version") or "unknown"),
        "tts_build_id": str(tts_status.get("build_id") or "unknown"),
        "tts_backend": str(tts_status.get("backend") or "unknown"),
        "hardware": str(tts_status.get("runtime") or "CPU/unknown"),
        "asr_model": asr_model,
        "minimum_similarity_percent": args.minimum_similarity,
        "comment": args.comment,
        "expected_text": EXPECTED_TEXT,
        "turn_endings": list(TURN_ENDINGS),
        "fixed_seeds": seeds,
        "documents": {scenario["id"]: scenario["document"] for scenario in SCENARIOS},
        "warmups": warmups,
        "summary": summary,
        "measurements": measurements,
    }

    if args.no_write:
        print("Benchmark completed without updating official history.", flush=True)
    else:
        append_json(args.results, run)
        append_summary(args.summary, run)
        append_details(args.details, run)

    for scenario in SCENARIOS:
        item = summary[scenario["id"]]
        print(
            f"{scenario['label']}: {item['passed']}/{item['rounds']} passed; "
            f"tail={item['tail_complete_percent']:.2f}%; endings={item['mean_turn_ending_percent']:.2f}%; "
            f"similarity={item['mean_similarity_percent']:.2f}%; median RTF={item['median_rtf']:.4f}",
            flush=True,
        )
    return 0 if all(item["passed"] == item["rounds"] for item in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
