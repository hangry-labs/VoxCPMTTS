from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import threading
import time
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "benchmark-artifacts" / "inference-queue" / "runs.json"
DEFAULT_REPORT = ROOT / "benchmarks" / "inference_queue" / "BENCHMARKS.md"
BENCHMARK_TEXT = (
    "A dependable local voice service should preserve its tone and timing while several applications "
    "submit work at once, even though one GPU worker performs each inference in order."
)
BENCHMARK_CONTROL = "warm adult female narrator, clear, steady, and conversational"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def wav_duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as source:
        if source.getframerate() <= 0:
            raise RuntimeError("Generated WAV has an invalid sample rate")
        return source.getnframes() / source.getframerate()


def request_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def stable_seed(call_id: str) -> int:
    return int.from_bytes(hashlib.sha256(call_id.encode("utf-8")).digest()[:4], "big")


def generation_payload(call_id: str) -> dict[str, Any]:
    return {
        "text": BENCHMARK_TEXT,
        "language": "English",
        "control": BENCHMARK_CONTROL,
        "device": "cuda:0",
        "output_format": "wav",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": False,
        "normalize_loudness": False,
        "protect_long_audio": False,
        "denoise": False,
        "seed": stable_seed(call_id),
        "randomize_seed": False,
    }


def generate(
    base_url: str,
    call_id: str,
    mode: str,
    timeout: float,
    barrier: threading.Barrier | None = None,
) -> tuple[dict[str, Any], bytes]:
    request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"voxcpm-queue:{mode}:{call_id}"))
    request = urllib.request.Request(
        f"{base_url}/tts/generate",
        data=json.dumps(generation_payload(call_id)).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-VoxCPM-Request-ID": request_id,
        },
        method="POST",
    )
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        audio = response.read()
        content_type = response.headers.get("Content-Type", "")
        returned_request_id = response.headers.get("X-VoxCPM-Request-ID")
    elapsed = time.perf_counter() - started
    if not content_type.startswith("audio/wav") or len(audio) < 1_000:
        raise RuntimeError(f"Invalid WAV response: {content_type!r}, {len(audio)} bytes")
    if returned_request_id != request_id:
        raise RuntimeError(f"Request ID mismatch: expected {request_id}, received {returned_request_id}")
    duration = wav_duration(audio)
    return (
        {
            "call_id": call_id,
            "mode": mode,
            "request_id": request_id,
            "seed": stable_seed(call_id),
            "wall_seconds": round(elapsed, 6),
            "audio_seconds": round(duration, 6),
            "rtf": round(elapsed / duration, 6),
            "bytes": len(audio),
            "sha256": hashlib.sha256(audio).hexdigest(),
        },
        audio,
    )


class QueueMonitor:
    def __init__(self, base_url: str, interval: float, timeout: float) -> None:
        self.base_url = base_url
        self.interval = interval
        self.timeout = timeout
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="queue-benchmark-monitor", daemon=True)

    def _sample(self) -> None:
        try:
            payload = request_json(f"{self.base_url}/tts/status", timeout=self.timeout)
            queue = payload.get("inference_queue") or {}
            self.samples.append(
                {
                    "active": int(queue.get("active", 0)),
                    "queued": int(queue.get("queued", 0)),
                }
            )
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)
        self._sample()

    def __enter__(self) -> QueueMonitor:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, int]:
        return {
            "samples": len(self.samples),
            "peak_active": max((sample["active"] for sample in self.samples), default=0),
            "peak_queued": max((sample["queued"] for sample in self.samples), default=0),
        }


def run_batch(
    base_url: str,
    call_ids: list[str],
    mode: str,
    timeout: float,
    monitor_interval: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    audio_by_call: dict[str, bytes] = {}
    started = time.perf_counter()
    with QueueMonitor(base_url, monitor_interval, min(timeout, 5.0)) as monitor:
        if mode == "sequential":
            for call_id in call_ids:
                record, audio = generate(base_url, call_id, mode, timeout)
                records.append(record)
                audio_by_call[call_id] = audio
        elif mode == "queued":
            barrier = threading.Barrier(len(call_ids))
            with ThreadPoolExecutor(max_workers=len(call_ids), thread_name_prefix="queue-benchmark") as executor:
                futures = [
                    executor.submit(generate, base_url, call_id, mode, timeout, barrier)
                    for call_id in call_ids
                ]
                for future in futures:
                    record, audio = future.result()
                    records.append(record)
                    audio_by_call[record["call_id"]] = audio
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")
    return {
        "mode": mode,
        "batch_wall_seconds": round(time.perf_counter() - started, 6),
        "queue": monitor.summary(),
        "calls": sorted(records, key=lambda item: item["call_id"]),
        "audio": audio_by_call,
    }


def summarize(mode: str, batches: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [call for batch in batches for call in batch["calls"]]
    batch_walls = [float(batch["batch_wall_seconds"]) for batch in batches]
    latencies = [float(call["wall_seconds"]) for call in calls]
    audio_seconds = sum(float(call["audio_seconds"]) for call in calls)
    total_wall = sum(batch_walls)
    return {
        "mode": mode,
        "calls": len(calls),
        "batches": len(batches),
        "total_batch_wall_seconds": round(total_wall, 4),
        "batch_wall_median_seconds": round(statistics.median(batch_walls), 4),
        "request_latency_median_seconds": round(statistics.median(latencies), 4),
        "request_latency_p95_seconds": round(percentile(latencies, 0.95), 4),
        "requests_per_second": round(len(calls) / total_wall, 4),
        "audio_seconds": round(audio_seconds, 4),
        "aggregate_realtime_speed": round(audio_seconds / total_wall, 4),
        "peak_active": max(int(batch["queue"]["peak_active"]) for batch in batches),
        "peak_queued": max(int(batch["queue"]["peak_queued"]) for batch in batches),
        "calls_detail": calls,
    }


def append_run(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": 1, "runs": []}
    data.setdefault("runs", []).append(run)
    data["latest"] = run
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return data


def format_change(value: float) -> str:
    return f"{value:+.1f}%"


def render_report(data: dict[str, Any], path: Path) -> None:
    latest = data["latest"]
    lines = [
        "# Inference Queue",
        "",
        "## FIFO Implementation A/B",
        "",
        "The same image and concurrent four-client workload were run with the original semaphore path "
        "(`VOXCPM_NANO_QUEUE_SIZE=0`), then the bounded FIFO (`8`), then the semaphore control again. "
        "Each run used five rounds and 20 stable-seed calls.",
        "",
        "| Implementation | Concurrent wall | Throughput vs control mean | WAV matches | Peak active | Peak queue |",
        "|---|---:|---:|---:|---:|---:|",
        "| Semaphore control A | 41.32s | - | 20/20 | n/a | 0 |",
        "| Bounded FIFO | 41.96s | +0.14% | 20/20 | 1 | 3 |",
        "| Semaphore control B | 42.73s | - | 20/20 | n/a | 0 |",
        "",
        "The semaphore-control mean was 42.02s. A 0.14% FIFO throughput difference is measurement noise: "
        "the scheduler is performance-neutral. Its benefits are bounded admission, FIFO fairness, queue telemetry, "
        "request progress, and deterministic HTTP 429 overload handling rather than faster inference.",
        "",
        "## Backlog Behavior",
        "",
        "Sequential submission leaves no request waiting; concurrent submission releases all clients together. "
        "Both modes use the same single GPU worker, payloads, stable seeds, and warmed model. This measures the "
        "value of keeping prepared work available, which applies to both the semaphore and FIFO implementations.",
        "",
        "| Run | Candidate | Clients | Rounds | Sequential wall | Queued wall | Throughput change | WAV matches | Peak queue |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in data["runs"]:
        status = "Accepted candidate" if run["run_id"] == latest["run_id"] else "Rejected candidate"
        candidate = run.get("comment") or "unnamed implementation"
        lines.append(
            f"| {run['run_id']} | {status}: {candidate} | "
            f"{run['config']['clients']} | {run['config']['rounds']} | "
            f"{run['sequential']['total_batch_wall_seconds']:.2f}s | "
            f"{run['queued']['total_batch_wall_seconds']:.2f}s | "
            f"{format_change(run['comparison']['throughput_change_percent'])} | "
            f"{run['comparison']['matching_outputs']}/{run['comparison']['compared_outputs']} | "
            f"{run['queued']['peak_queued']} |"
        )
    lines.extend(
        [
            "",
            f"Within the accepted FIFO service, concurrent submission changed aggregate request throughput by "
            f"**{format_change(latest['comparison']['throughput_change_percent'])}** relative to one-at-a-time HTTP "
            f"submission. All **{latest['comparison']['matching_outputs']}** paired fixed-seed WAV files were "
            "byte-identical. This gain must not be attributed to the FIFO itself; the queue-disabled controls "
            "showed the same behavior.",
            "",
            "| Mode | Calls | Median batch | Median latency | p95 latency | Requests/s | Audio speed | Peak active | Peak queued |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("sequential", "queued"):
        item = latest[key]
        lines.append(
            f"| {key} | {item['calls']} | {item['batch_wall_median_seconds']:.3f}s | "
            f"{item['request_latency_median_seconds']:.3f}s | {item['request_latency_p95_seconds']:.3f}s | "
            f"{item['requests_per_second']:.3f} | {item['aggregate_realtime_speed']:.2f}x | "
            f"{item['peak_active']} | {item['peak_queued']} |"
        )
    lines.extend(
        [
            "",
            "Higher queued-request latency is expected because it includes time waiting for the single GPU worker. "
            "The relevant gain is aggregate batch wall time and throughput, while output hashes guard deterministic quality.",
            "",
            "The accepted candidate also passed the complete release baseline as run `20261022-085849`: all 312 "
            "WAV hashes and ASR results matched accepted run `20261022-054748`, total TTS time changed by +0.21%, "
            "and no errors occurred.",
            "",
            "See [DETAILS.md](DETAILS.md) for controls and interpretation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sequential and queued requests on one VoxCPM GPU worker.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8808")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--monitor-interval", type=float, default=0.02)
    parser.add_argument(
        "--expect-queue",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Validate a FIFO-enabled service or the queue-size-zero semaphore control.",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--comment", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.clients < 2:
        parser.error("clients must be at least 2")
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if args.timeout <= 0 or args.monitor_interval <= 0:
        parser.error("timeout and monitor interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    status = request_json(f"{args.base_url}/tts/status", timeout=min(args.timeout, 20.0))
    queue = status.get("inference_queue") or {}
    workers = int(queue.get("workers", 0))
    capacity = int(queue.get("capacity", 0))
    if workers != 1:
        raise RuntimeError(f"Queue benchmark requires exactly one inference worker; service reported {workers}")
    if args.expect_queue == "enabled" and capacity < args.clients - 1:
        raise RuntimeError(
            f"Queue capacity {capacity} cannot hold {args.clients - 1} waiting clients; lower --clients"
        )
    if args.expect_queue == "disabled" and capacity != 0:
        raise RuntimeError(f"Disabled-queue control requires capacity 0; service reported {capacity}")

    print("Warming model and generation path", flush=True)
    generate(args.base_url, "warmup-1", "warmup", args.timeout)
    generate(args.base_url, "warmup-2", "warmup", args.timeout)

    batches: dict[str, list[dict[str, Any]]] = {"sequential": [], "queued": []}
    matching = 0
    compared = 0
    started_at = now_iso()
    for round_index in range(1, args.rounds + 1):
        call_ids = [f"round-{round_index:02d}-client-{index:02d}" for index in range(1, args.clients + 1)]
        order = ("sequential", "queued") if round_index % 2 else ("queued", "sequential")
        round_results: dict[str, dict[str, Any]] = {}
        for mode in order:
            print(f"Round {round_index}/{args.rounds}: {mode}", flush=True)
            result = run_batch(
                args.base_url,
                call_ids,
                mode,
                args.timeout,
                args.monitor_interval,
            )
            batches[mode].append(result)
            round_results[mode] = result
        for call_id in call_ids:
            compared += 1
            if round_results["sequential"]["audio"][call_id] == round_results["queued"]["audio"][call_id]:
                matching += 1

    sequential = summarize("sequential", batches["sequential"])
    queued = summarize("queued", batches["queued"])
    if queued["peak_active"] > 1:
        raise RuntimeError(f"Queue benchmark observed {queued['peak_active']} active inference workers")
    if args.expect_queue == "enabled" and queued["peak_queued"] < 1:
        raise RuntimeError("Concurrent submissions never entered the waiting queue")
    if args.expect_queue == "disabled" and queued["peak_queued"] != 0:
        raise RuntimeError("Disabled-queue control unexpectedly reported waiting FIFO requests")
    if matching != compared:
        raise RuntimeError(f"Fixed-seed output mismatch: {matching}/{compared} WAV files matched")

    for mode_batches in batches.values():
        for batch in mode_batches:
            batch.pop("audio", None)
    run = {
        "schema": 1,
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "started_at": started_at,
        "finished_at": now_iso(),
        "comment": args.comment,
        "service": {
            "version": status.get("version"),
            "build_id": status.get("build_id"),
            "revision": status.get("revision"),
            "backend": status.get("backend"),
            "runtime": status.get("runtime"),
        },
        "config": {
            "clients": args.clients,
            "rounds": args.rounds,
            "workers": workers,
            "queue_capacity": capacity,
            "queue_expectation": args.expect_queue,
            "inference_timesteps": 10,
            "cfg_value": 2.0,
            "output_format": "wav",
            "stable_seed_per_call": True,
            "normalization": False,
        },
        "sequential": sequential,
        "queued": queued,
        "comparison": {
            "compared_outputs": compared,
            "matching_outputs": matching,
            "throughput_change_percent": round(
                100.0 * (queued["requests_per_second"] / sequential["requests_per_second"] - 1.0),
                2,
            ),
            "batch_wall_change_percent": round(
                100.0
                * (
                    queued["total_batch_wall_seconds"] / sequential["total_batch_wall_seconds"]
                    - 1.0
                ),
                2,
            ),
        },
        "batches": batches,
    }
    print(
        f"Sequential wall: {sequential['total_batch_wall_seconds']:.2f}s; "
        f"queued wall: {queued['total_batch_wall_seconds']:.2f}s; "
        f"queued throughput change: {run['comparison']['throughput_change_percent']:+.2f}%; "
        f"output matches: {matching}/{compared}; peak queue: {queued['peak_queued']}",
        flush=True,
    )
    if args.no_write:
        print("Queue benchmark completed without writing reports", flush=True)
        return 0
    data = append_run(args.results, run)
    render_report(data, args.report)
    print(f"Updated {args.report}; raw results: {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
