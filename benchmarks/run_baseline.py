from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import math
import shutil
import statistics
import sys
import tempfile
import time
import unicodedata
import urllib.request
import uuid
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.backend_comparison.run_benchmark import (  # noqa: E402
    CONTAINER_REFERENCE_PATH,
    GpuSampler,
    container_memory,
    container_running,
    docker,
    gpu_snapshot,
    image_metadata,
    remove_container,
    request_json,
    runtime_versions,
    start_container,
    wait_for_gpu_settle,
)


MANIFEST_PATH = ROOT / "examples" / "assets" / "manifest.json"
REFERENCE_PATH = ROOT / "examples" / "original_clone.mp3"
ARTIFACTS_ROOT = ROOT / "benchmark-artifacts" / "speech-quality"
SUITES = {
    "speed": ROOT / "benchmarks" / "speed",
    "memory": ROOT / "benchmarks" / "memory" / "gpu",
    "quality": ROOT / "benchmarks" / "speech-quality",
}
LANGUAGE_ALIASES = {"tagalog": "Filipino", "standard arabic": "Arabic"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if unicodedata.category(character)[0] in {"L", "N"})


def similarity(expected: str, actual: str) -> float:
    left = normalize_transcript(expected)
    right = normalize_transcript(actual)
    if not left and not right:
        return 100.0
    return round(100.0 * difflib.SequenceMatcher(None, left, right).ratio(), 2)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wav_duration(audio: bytes) -> tuple[float, int]:
    with wave.open(io.BytesIO(audio), "rb") as source:
        return source.getnframes() / source.getframerate(), source.getframerate()


def asr_languages(asr_url: str) -> tuple[set[str], dict[str, Any], list[dict[str, Any]]]:
    health = request_json("GET", f"{asr_url}/health/ready", timeout=20)
    payload = request_json("GET", f"{asr_url}/v1/audio/supported_languages", timeout=20)
    models = request_json("GET", f"{asr_url}/v1/models", timeout=20).get("data", [])
    return {str(item) for item in payload.get("languages", [])}, health, models


def resolved_asr_language(language: str, supported: set[str]) -> str | None:
    candidate = LANGUAGE_ALIASES.get(language.casefold(), language)
    lookup = {item.casefold(): item for item in supported}
    return lookup.get(candidate.casefold())


def load_workload(
    manifest_path: Path,
    supported: set[str],
    languages_filter: set[str],
    design_sentences: int,
    repeats: int,
    clone_repeats: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workload: list[dict[str, Any]] = []
    selected: list[str] = []
    for language_entry in manifest["languages"]:
        language = str(language_entry["language"])
        slug = str(language_entry["slug"])
        asr_language = resolved_asr_language(language, supported)
        if asr_language is None:
            continue
        if languages_filter and language.casefold() not in languages_filter and slug.casefold() not in languages_filter:
            continue
        selected.append(language)
        random_items = language_entry.get("random", [])
        if len(random_items) < design_sentences:
            raise ValueError(f"{language} has only {len(random_items)} design sentences")
        for sentence_index, item in enumerate(random_items[:design_sentences], start=1):
            settings = item.get("settings") or {}
            for repeat in range(1, repeats + 1):
                workload.append(
                    {
                        "case_id": f"{slug}_design_{sentence_index:02d}",
                        "call_id": f"{slug}_design_{sentence_index:02d}_r{repeat}",
                        "mode": "design",
                        "language": language,
                        "language_slug": slug,
                        "asr_language": asr_language,
                        "text": str(item["text"]),
                        "control": item.get("control") or settings.get("control"),
                        "repeat": repeat,
                    }
                )
        clone_items = language_entry.get("clone") or []
        if clone_repeats and clone_items:
            for repeat in range(1, clone_repeats + 1):
                workload.append(
                    {
                        "case_id": f"{slug}_clone_01",
                        "call_id": f"{slug}_clone_01_r{repeat}",
                        "mode": "clone",
                        "language": language,
                        "language_slug": slug,
                        "asr_language": asr_language,
                        "text": str(clone_items[0]["text"]),
                        "control": None,
                        "repeat": repeat,
                    }
                )
    if not selected:
        raise RuntimeError("No common VoxCPM and Qwen3-ASR languages matched the requested filter")
    return workload, selected


def tts_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "text": item["text"],
        "language": item["language"],
        "device": "cuda:0",
        "output_format": "wav",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": False,
        "denoise": False,
    }
    if item["mode"] == "clone":
        payload["ref_audio"] = CONTAINER_REFERENCE_PATH
    elif item.get("control"):
        payload["control"] = item["control"]
    return payload


def generate(base_url: str, item: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url}/tts/generate",
        data=json.dumps(tts_payload(item), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        audio = response.read()
        content_type = response.headers.get("Content-Type", "")
    elapsed = time.perf_counter() - started
    if not content_type.startswith("audio/wav") or len(audio) < 1_000:
        raise RuntimeError(f"Invalid WAV response: {content_type!r}, {len(audio)} bytes")
    duration, sample_rate = wav_duration(audio)
    return audio, {
        "tts_seconds": round(elapsed, 4),
        "audio_seconds": round(duration, 4),
        "rtf": round(elapsed / duration, 4),
        "bytes": len(audio),
        "sample_rate": sample_rate,
        "sha256": hashlib.sha256(audio).hexdigest(),
    }


def multipart_body(audio: bytes, language: str, filename: str) -> tuple[bytes, str]:
    boundary = f"----VoxCPMBenchmark{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in (
        ("model", "qwen3-asr"),
        ("response_format", "verbose_json"),
        ("temperature", "0"),
        ("language", language),
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


def transcribe(asr_url: str, audio: bytes, language: str, filename: str) -> tuple[dict[str, Any], float]:
    body, boundary = multipart_body(audio, language, filename)
    request = urllib.request.Request(
        f"{asr_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, time.perf_counter() - started


def append_run(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": 1, "runs": []}
    data.setdefault("runs", []).append(run)
    data["latest"] = run
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return data


def summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    tts_good = [item for item in calls if "tts_seconds" in item]
    quality_good = [item for item in tts_good if "similarity_percent" in item]
    rtfs = [float(item["rtf"]) for item in tts_good]
    similarities = [float(item["similarity_percent"]) for item in quality_good]
    exact = sum(bool(item.get("exact")) for item in quality_good)
    gpu_peaks = [
        int(item["gpu"]["peak_delta_mib"])
        for item in tts_good
        if item.get("gpu", {}).get("peak_delta_mib") is not None
    ]
    total_tts_seconds = sum(float(item["tts_seconds"]) for item in tts_good)
    return {
        "calls": len(calls),
        "completed": len(tts_good),
        "quality_completed": len(quality_good),
        "tts_errors": len(calls) - len(tts_good),
        "asr_errors": len(tts_good) - len(quality_good),
        "errors": len(calls) - len(quality_good),
        "audio_seconds": round(sum(float(item["audio_seconds"]) for item in tts_good), 3),
        "tts_seconds": round(total_tts_seconds, 3),
        "rtf_median": round(statistics.median(rtfs), 4) if rtfs else None,
        "rtf_p95": round(percentile(rtfs, 0.95), 4) if rtfs else None,
        "realtime_speed": round(sum(float(item["audio_seconds"]) for item in tts_good) / total_tts_seconds, 2)
        if total_tts_seconds
        else None,
        "exact_percent": round(100 * exact / len(quality_good), 2) if quality_good else 0.0,
        "mean_similarity_percent": round(statistics.fmean(similarities), 2) if similarities else 0.0,
        "min_similarity_percent": round(min(similarities), 2) if similarities else 0.0,
        "gpu_peak_delta_mib": max(gpu_peaks) if gpu_peaks else None,
    }


def consistency(calls: list[dict[str, Any]]) -> tuple[int, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in calls:
        grouped[item["case_id"]].append(item)
    consistent = 0
    for values in grouped.values():
        transcripts = {item.get("transcript_normalized") for item in values if not item.get("error")}
        if len(values) > 1 and len(transcripts) == 1 and None not in transcripts:
            consistent += 1
    repeated = sum(1 for values in grouped.values() if len(values) > 1)
    return consistent, repeated


def backend_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {mode: summarize_calls([item for item in calls if item["mode"] == mode]) for mode in ("design", "clone")}
    by_language = {
        language: summarize_calls([item for item in calls if item["language"] == language])
        for language in sorted({item["language"] for item in calls})
    }
    consistent, repeated = consistency(calls)
    return {
        "overall": summarize_calls(calls),
        "by_mode": by_mode,
        "by_language": by_language,
        "consistent_cases": consistent,
        "repeated_cases": repeated,
        "consistent_case_percent": round(100 * consistent / repeated, 2) if repeated else 0.0,
    }


def run_backend(
    args: argparse.Namespace,
    backend: str,
    image: str,
    workload: list[dict[str, Any]],
) -> dict[str, Any]:
    args.active_backend = backend
    print(f"\n[{backend}] starting {image}", flush=True)
    ready_seconds = start_container(args, image)
    try:
        status_before = request_json("GET", f"{args.base_url}/tts/status")
        reported = status_before.get("backend")
        if backend == "nano" and reported != "nano":
            raise RuntimeError(f"Nano image reported backend {reported!r}")
        baseline = gpu_snapshot(args.gpu_index)
        baseline_mib = int(baseline["used_mib"]) if baseline else None

        warmups = [next(item for item in workload if item["mode"] == "design")]
        if any(item["mode"] == "clone" for item in workload):
            warmups.append(next(item for item in workload if item["mode"] == "clone"))
        for item in warmups:
            print(f"[{backend}] warmup {item['mode']}", flush=True)
            with GpuSampler(args.gpu_index, args.sample_interval):
                generate(args.base_url, item)

        calls: list[dict[str, Any]] = []
        run_id = args.run_id
        with tempfile.TemporaryDirectory(prefix=f"voxcpm-{backend}-") as temporary:
            staging = Path(temporary)
            print(f"[{backend}] phase 1/2: {len(workload)} TTS calls", flush=True)
            for index, item in enumerate(workload, start=1):
                result = dict(item)
                try:
                    with GpuSampler(args.gpu_index, args.sample_interval) as sampler:
                        audio, metrics = generate(args.base_url, item)
                    result.update(metrics)
                    result["gpu"] = sampler.summary(baseline_mib)
                    output = staging / f"{item['call_id']}.wav"
                    output.write_bytes(audio)
                    result["staged_audio"] = str(output)
                    outcome = f"RTF {result['rtf']:.3f}"
                except Exception as exc:
                    result.update({"error_stage": "tts", "error": f"{type(exc).__name__}: {exc}"})
                    outcome = "ERROR"
                calls.append(result)
                print(f"  [{index:03d}/{len(workload):03d}] {item['call_id']}: {outcome}", flush=True)

            asr_warmup = next((item for item in calls if item.get("staged_audio")), None)
            if asr_warmup:
                transcribe(
                    args.asr_url,
                    Path(asr_warmup["staged_audio"]).read_bytes(),
                    asr_warmup["asr_language"],
                    "warmup",
                )
            print(f"[{backend}] phase 2/2: Qwen3-ASR judging", flush=True)
            for index, result in enumerate(calls, start=1):
                if result.get("error"):
                    continue
                audio = Path(result["staged_audio"]).read_bytes()
                try:
                    payload, elapsed = transcribe(args.asr_url, audio, result["asr_language"], result["call_id"])
                    transcript = str(payload.get("text") or "").strip()
                    result.update(
                        {
                            "transcript": transcript,
                            "transcript_normalized": normalize_transcript(transcript),
                            "detected_language": payload.get("language"),
                            "asr_seconds": round(elapsed, 4),
                            "exact": normalize_transcript(result["text"]) == normalize_transcript(transcript),
                            "similarity_percent": similarity(result["text"], transcript),
                        }
                    )
                    outcome = "MATCH" if result["exact"] else f"{result['similarity_percent']:.1f}%"
                    if not result["exact"]:
                        artifact = ARTIFACTS_ROOT / run_id / backend / f"{result['call_id']}.wav"
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(result["staged_audio"], artifact)
                        result["artifact"] = str(artifact.relative_to(ROOT))
                except Exception as exc:
                    result.update({"error_stage": "asr", "error": f"{type(exc).__name__}: {exc}"})
                    artifact = ARTIFACTS_ROOT / run_id / backend / f"{result['call_id']}.wav"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(result["staged_audio"], artifact)
                    result["artifact"] = str(artifact.relative_to(ROOT))
                    outcome = "ERROR"
                result.pop("staged_audio", None)
                print(f"  [{index:03d}/{len(calls):03d}] {result['call_id']}: {outcome}", flush=True)

        final_gpu = gpu_snapshot(args.gpu_index)
        summary = backend_summary(calls)
        return {
            "backend": backend,
            "image": image_metadata(image),
            "runtime_versions": runtime_versions(args.container),
            "api_ready_seconds": round(ready_seconds, 3),
            "status_before": status_before,
            "status_after": request_json("GET", f"{args.base_url}/tts/status"),
            "gpu_baseline": baseline,
            "gpu_final": final_gpu,
            "gpu_final_delta_mib": int(final_gpu["used_mib"]) - baseline_mib
            if final_gpu is not None and baseline_mib is not None
            else None,
            "container_memory": container_memory(args.container),
            "summary": summary,
            "calls": calls,
        }
    finally:
        remove_container(args.container)


def backend_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["backend"]: item for item in run["backends"]}


def format_metric(value: Any, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}{suffix}"


def render_speed(data: dict[str, Any]) -> None:
    run = data["latest"]
    backends = backend_map(run)
    lines = [
        "# Inference Speed",
        "",
        "Warmed multilingual API generation. Lower real-time factor (RTF) is better.",
        "",
        "| Run | Backend | Calls | Audio | TTS time | Median RTF | p95 RTF | Realtime speed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for historical in data["runs"]:
        for item in historical["backends"]:
            summary = item["summary"]["overall"]
            lines.append(
                f"| {historical['run_id']} | {item['backend']} | {summary['completed']} | {summary['audio_seconds']:.1f}s | "
                f"{summary['tts_seconds']:.1f}s | {format_metric(summary['rtf_median'], 3)} | "
                f"{format_metric(summary['rtf_p95'], 3)} | {format_metric(summary['realtime_speed'], 2, 'x')} |"
            )
    if {"native", "nano"} <= set(backends):
        native = backends["native"]["summary"]["overall"]["rtf_median"]
        nano = backends["nano"]["summary"]["overall"]["rtf_median"]
        if native is not None and nano:
            lines.extend(["", f"Latest median Nano speedup: **{native / nano:.2f}x**."])
    lines.extend(["", "See [DETAILS.md](DETAILS.md) for per-language and per-mode results.", ""])
    (SUITES["speed"] / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")

    details = [
        "# Inference Speed Details",
        "",
        f"Run `{run['run_id']}`; 10 inference steps; concurrency 1; warmed requests.",
        "RTF is wall-clock generation time divided by generated audio duration. Lower is better.",
        "Generation is unseeded because the VoxCPM API does not expose a shared deterministic seed.",
        "",
    ]
    for backend in run["backends"]:
        details.extend([f"## {backend['backend']}", "", "| Mode | Calls | Median RTF | p95 RTF | Speed |", "|---|---:|---:|---:|---:|"])
        for mode, row in backend["summary"]["by_mode"].items():
            details.append(
                f"| {mode} | {row['completed']} | {format_metric(row['rtf_median'], 3)} | "
                f"{format_metric(row['rtf_p95'], 3)} | {format_metric(row['realtime_speed'], 2, 'x')} |"
            )
        details.extend(["", "### Languages", "", "| Language | Calls | Median RTF | p95 RTF | Speed |", "|---|---:|---:|---:|---:|"])
        for language, row in backend["summary"]["by_language"].items():
            details.append(
                f"| {language} | {row['completed']} | {format_metric(row['rtf_median'], 3)} | "
                f"{format_metric(row['rtf_p95'], 3)} | {format_metric(row['realtime_speed'], 2, 'x')} |"
            )
        details.append("")
    (SUITES["speed"] / "DETAILS.md").write_text("\n".join(details), encoding="utf-8")


def render_memory(data: dict[str, Any]) -> None:
    run = data["latest"]
    lines = [
        "# GPU Memory",
        "",
        "Whole-device VRAM deltas subtract the pre-load baseline. Container RAM uses cgroup counters.",
        "",
        "| Run | Backend | Baseline VRAM | Peak delta | Final delta | RAM current | RAM peak |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for historical in data["runs"]:
        for item in historical["backends"]:
            summary = item["summary"]["overall"]
            ram = item["container_memory"]
            lines.append(
                f"| {historical['run_id']} | {item['backend']} | {item['gpu_baseline']['used_mib']} MiB | "
                f"{summary['gpu_peak_delta_mib']} MiB | {item['gpu_final_delta_mib']} MiB | "
                f"{ram['current_mib']} MiB | {ram['peak_mib']} MiB |"
            )
    lines.append("")
    (SUITES["memory"] / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    details = [
        "# GPU Memory Details",
        "",
        f"Run `{run['run_id']}` on `{run['hardware']['name']}`.",
        "VRAM is sampled every 100 ms from the whole selected GPU and reported relative to the pre-load baseline.",
        "",
    ]
    for item in run["backends"]:
        details.extend([f"## {item['backend']}", "", "| Mode | Calls | Peak VRAM delta |", "|---|---:|---:|"])
        for mode, row in item["summary"]["by_mode"].items():
            peak = f"{row['gpu_peak_delta_mib']} MiB" if row["gpu_peak_delta_mib"] is not None else "n/a"
            details.append(f"| {mode} | {row['completed']} | {peak} |")
        details.append("")
    (SUITES["memory"] / "DETAILS.md").write_text("\n".join(details), encoding="utf-8")


def render_quality(data: dict[str, Any]) -> None:
    run = data["latest"]
    lines = [
        "# Speech Quality",
        "",
        "Qwen3-ASR is a fixed comparative semantic judge, not ground truth or a replacement for listening tests.",
        "",
        "| Run | Backend | Calls | Exact | Mean similarity | Minimum | Repeat consistency | Errors | ASR model |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for historical in data["runs"]:
        for item in historical["backends"]:
            row = item["summary"]["overall"]
            lines.append(
                f"| {historical['run_id']} | {item['backend']} | {row['quality_completed']} | {row['exact_percent']:.2f}% | "
                f"{row['mean_similarity_percent']:.2f}% | {row['min_similarity_percent']:.2f}% | "
                f"{item['summary']['consistent_case_percent']:.2f}% | {row['errors']} | {historical['asr']['model']} |"
            )
    lines.extend(["", "Non-exact WAV files are retained locally under the git-ignored `benchmark-artifacts/` directory.", ""])
    (SUITES["quality"] / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    details = [
        "# Speech Quality Details",
        "",
        f"Run `{run['run_id']}` across {len(run['languages'])} common languages.",
        "Text is normalized with Unicode NFKC, case folding, and removal of non-letter/non-number characters.",
        "Similarity uses a normalized sequence ratio; exact means the normalized transcript equals the prompt.",
        "Generation is stochastic; repeated samples characterize the distribution rather than paired identical outputs.",
        "",
    ]
    for item in run["backends"]:
        details.extend([f"## {item['backend']}", "", "| Mode | Judged | Exact | Mean similarity | Minimum | Errors |", "|---|---:|---:|---:|---:|---:|"])
        for mode, row in item["summary"]["by_mode"].items():
            details.append(
                f"| {mode} | {row['quality_completed']} | {row['exact_percent']:.2f}% | "
                f"{row['mean_similarity_percent']:.2f}% | {row['min_similarity_percent']:.2f}% | {row['errors']} |"
            )
        details.extend(["", "### Languages", "", "| Language | Judged | Exact | Mean similarity | Minimum | Errors |", "|---|---:|---:|---:|---:|---:|"])
        for language, row in item["summary"]["by_language"].items():
            details.append(f"| {language} | {row['quality_completed']} | {row['exact_percent']:.2f}% | {row['mean_similarity_percent']:.2f}% | {row['min_similarity_percent']:.2f}% | {row['errors']} |")
        hardest = sorted((call for call in item["calls"] if not call.get("error")), key=lambda call: call.get("similarity_percent", 0))[:20]
        details.extend(["", "### Hardest Calls", "", "| Call | Expected | Transcript | Similarity |", "|---|---|---|---:|"])
        for call in hardest:
            expected = call["text"].replace("|", "/")
            transcript = call.get("transcript", "").replace("|", "/")
            details.append(f"| {call['call_id']} | {expected} | {transcript} | {call['similarity_percent']:.2f}% |")
        details.append("")
    (SUITES["quality"] / "DETAILS.md").write_text("\n".join(details), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled VoxCPM native/Nano multilingual baseline.")
    parser.add_argument("--native-image", default="voxcpmtts:benchmark-native")
    parser.add_argument("--nano-image", default="voxcpmtts:benchmark-nano")
    parser.add_argument("--container", default="voxcpmtts_benchmark")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cache-volume", default="voxcpmtts_hf_cache")
    parser.add_argument("--reference-audio", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--asr-url", default="http://127.0.0.1:8000")
    parser.add_argument("--design-sentences", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--clone-repeats", type=int, default=3)
    parser.add_argument("--languages", default="")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--stop-container", action="append", default=[])
    parser.add_argument("--comment", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if min(args.design_sentences, args.repeats) < 1 or args.clone_repeats < 0:
        parser.error("design sentences and repeats must be positive; clone repeats must not be negative")
    args.base_url = f"http://127.0.0.1:{args.port}"
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return args


def main() -> int:
    args = parse_args()
    supported, asr_health, models = asr_languages(args.asr_url)
    filters = {item.strip().casefold() for item in args.languages.split(",") if item.strip()}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    vox_languages = [str(item["language"]) for item in manifest["languages"]]
    workload, languages = load_workload(
        args.manifest,
        supported,
        filters,
        args.design_sentences,
        args.repeats,
        args.clone_repeats,
    )
    hardware = gpu_snapshot(args.gpu_index)
    if hardware is None:
        raise RuntimeError(f"Cannot inspect GPU {args.gpu_index}")
    images = {"native": args.native_image, "nano": args.nano_image}
    for image in images.values():
        image_metadata(image)
    docker("volume", "inspect", args.cache_volume)

    print(
        f"Common language set: {len(languages)} languages; {len(workload)} measured calls per backend",
        flush=True,
    )
    previously_running = []
    for name in dict.fromkeys(args.stop_container):
        if name != args.container and container_running(name):
            docker("stop", "--time", "60", name, timeout=90)
            previously_running.append(name)

    backends = []
    started_at = now_iso()
    try:
        settle_target = int((gpu_snapshot(args.gpu_index) or hardware)["used_mib"]) + 128
        for backend, image in images.items():
            wait_for_gpu_settle(args.gpu_index, settle_target)
            backends.append(run_backend(args, backend, image, workload))
    finally:
        remove_container(args.container)
        for name in previously_running:
            docker("start", name, check=False, timeout=90)

    model = str(models[0].get("id")) if models else str(asr_health.get("model") or "unknown")
    run = {
        "schema": 1,
        "run_id": args.run_id,
        "started_at": started_at,
        "finished_at": now_iso(),
        "comment": args.comment,
        "hardware": hardware,
        "config": {
            "model_id": "openbmb/VoxCPM2",
            "inference_timesteps": 10,
            "cfg_value": 2.0,
            "concurrency": 1,
            "design_sentences_per_language": args.design_sentences,
            "design_repeats": args.repeats,
            "clone_repeats": args.clone_repeats,
            "sample_interval_seconds": args.sample_interval,
            "seed_policy": "unseeded; VoxCPM API does not expose a shared deterministic seed",
        },
        "languages": languages,
        "language_selection": {
            "voxcpm_languages": vox_languages,
            "common_languages": languages,
            "excluded_without_asr_support": [
                language for language in vox_languages if resolved_asr_language(language, supported) is None
            ],
        },
        "asr": {
            "model": model,
            "role": "fixed comparative semantic judge; not ground truth",
            "supported_languages": sorted(supported),
            "endpoint": "private service" if "127.0.0.1" not in args.asr_url and "localhost" not in args.asr_url else "local service",
        },
        "backends": backends,
    }
    if args.no_write:
        print("Baseline smoke completed; no history was written.", flush=True)
        return 0
    datasets = {name: append_run(directory / "runs.json", run) for name, directory in SUITES.items()}
    render_speed(datasets["speed"])
    render_memory(datasets["memory"])
    render_quality(datasets["quality"])
    print("Updated speed, GPU-memory, and speech-quality reports.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
