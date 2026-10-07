from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "benchmarks" / "backend_comparison" / "runs.json"
SUMMARY_PATH = ROOT / "benchmarks" / "backend_comparison" / "BENCHMARKS.md"
REFERENCE_PATH = ROOT / "examples" / "original_clone.mp3"
CONTAINER_REFERENCE_PATH = "/benchmark/original_clone.mp3"


@dataclass(frozen=True)
class Scenario:
    code: str
    label: str
    description: str
    payload: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def command(
    args: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=timeout)


def docker(*args: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return command(["docker", *args], check=check, timeout=timeout)


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(base_url: str, timeout: int = 180) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        try:
            if request_json("GET", f"{base_url}/tts/ping", timeout=5).get("msg") == "pong":
                return time.perf_counter() - started
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise RuntimeError(f"API did not become ready at {base_url} within {timeout} seconds")


def wav_duration(audio: bytes) -> tuple[float, int, int]:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
    if sample_rate <= 0:
        raise RuntimeError("Generated WAV reported an invalid sample rate")
    return frames / sample_rate, sample_rate, channels


def generate_audio(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/tts/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        audio = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
        content_type = response.headers.get("Content-Type", "")
    elapsed = time.perf_counter() - started
    if not content_type.startswith("audio/wav") or len(audio) < 1_000:
        raise RuntimeError(f"Invalid WAV response: {content_type!r}, {len(audio)} bytes")
    duration, sample_rate, channels = wav_duration(audio)
    if duration <= 0:
        raise RuntimeError("Generated WAV was empty")
    return {
        "wall_seconds": round(elapsed, 4),
        "audio_seconds": round(duration, 4),
        "rtf": round(elapsed / duration, 4),
        "bytes": len(audio),
        "sample_rate": sample_rate,
        "channels": channels,
        "response_duration": headers.get("x-voxcpm-duration"),
        "response_model": headers.get("x-voxcpm-model"),
    }


def gpu_snapshot(gpu_index: int) -> dict[str, Any] | None:
    args = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = command(args, timeout=15)
        line = next(line for line in result.stdout.splitlines() if line.strip())
        index, name, driver, total, used, utilization = (part.strip() for part in line.split(",", 5))
        return {
            "index": int(index),
            "name": name,
            "driver": driver,
            "total_mib": int(total),
            "used_mib": int(used),
            "utilization_percent": int(utilization),
        }
    except (FileNotFoundError, StopIteration, subprocess.SubprocessError, ValueError):
        return None


class GpuSampler:
    def __init__(self, gpu_index: int, interval: float) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        snapshot = gpu_snapshot(self.gpu_index)
        if snapshot is not None:
            self.samples.append(snapshot)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)
        self._sample()

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15)

    def summary(self, baseline_mib: int | None) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0, "peak_mib": None, "peak_delta_mib": None, "mean_utilization_percent": None}
        used = [int(item["used_mib"]) for item in self.samples]
        utilization = [int(item["utilization_percent"]) for item in self.samples]
        peak = max(used)
        return {
            "samples": len(self.samples),
            "peak_mib": peak,
            "peak_delta_mib": peak - baseline_mib if baseline_mib is not None else None,
            "mean_utilization_percent": round(statistics.fmean(utilization), 1),
            "peak_utilization_percent": max(utilization),
        }


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile without values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(item["wall_seconds"]) for item in measurements]
    rtfs = [float(item["rtf"]) for item in measurements]
    durations = [float(item["audio_seconds"]) for item in measurements]
    peaks = [int(item["gpu"]["peak_mib"]) for item in measurements if item["gpu"]["peak_mib"] is not None]
    deltas = [int(item["gpu"]["peak_delta_mib"]) for item in measurements if item["gpu"]["peak_delta_mib"] is not None]
    utils = [float(item["gpu"]["mean_utilization_percent"]) for item in measurements if item["gpu"]["mean_utilization_percent"] is not None]
    return {
        "calls": len(measurements),
        "wall_median_seconds": round(statistics.median(walls), 4),
        "wall_p95_seconds": round(percentile(walls, 0.95), 4),
        "wall_mean_seconds": round(statistics.fmean(walls), 4),
        "rtf_median": round(statistics.median(rtfs), 4),
        "rtf_p95": round(percentile(rtfs, 0.95), 4),
        "rtf_mean": round(statistics.fmean(rtfs), 4),
        "audio_mean_seconds": round(statistics.fmean(durations), 4),
        "gpu_peak_mib": max(peaks) if peaks else None,
        "gpu_peak_delta_mib": max(deltas) if deltas else None,
        "gpu_mean_utilization_percent": round(statistics.fmean(utils), 1) if utils else None,
    }


def build_scenarios(reference_audio: str) -> list[Scenario]:
    base = {
        "language": "English",
        "device": "cuda:0",
        "output_format": "wav",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": False,
        "denoise": False,
    }
    return [
        Scenario(
            "PS",
            "plain_short_english",
            "Short English generation without voice control or reference audio.",
            {**base, "text": "The morning coffee negotiated hard, but eventually agreed to help."},
        ),
        Scenario(
            "DM",
            "designed_medium_english",
            "Medium English generation with a fixed voice-design instruction.",
            {
                **base,
                "text": (
                    "Local speech generation keeps private text on your own machine while producing clear, "
                    "natural audio for applications, accessibility tools, and creative work."
                ),
                "control": "warm adult female narrator, clear and friendly",
            },
        ),
        Scenario(
            "MN",
            "designed_norwegian",
            "Norwegian generation with the same fixed voice-design instruction.",
            {
                **base,
                "language": "Norwegian",
                "text": "Denne testen måler hvor raskt modellen lager tydelig norsk tale med de samme innstillingene.",
                "control": "warm adult female narrator, clear and friendly",
            },
        ),
        Scenario(
            "CR",
            "reference_clone",
            "English zero-shot clone with the same reference audio mounted into both containers.",
            {
                **base,
                "text": "This sample checks how quickly the model can carry a reference voice into a new sentence.",
                "ref_audio": reference_audio,
            },
        ),
    ]


def image_metadata(image: str) -> dict[str, Any]:
    payload = json.loads(docker("image", "inspect", image, timeout=30).stdout)[0]
    return {
        "tag": image,
        "id": payload.get("Id"),
        "created": payload.get("Created"),
        "size_bytes": payload.get("Size"),
        "repo_digests": payload.get("RepoDigests") or [],
        "labels": (payload.get("Config") or {}).get("Labels") or {},
    }


def runtime_versions(container: str) -> dict[str, str | None]:
    script = (
        "import importlib.metadata as m,json,sys,torch;"
        "names=['transformers','huggingface-hub','nano-vllm-voxcpm','flash-attn','triton'];"
        "versions={n:(m.version(n) if any(d.metadata.get('Name','').lower()==n for d in m.distributions()) else None) for n in names};"
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,**versions}))"
    )
    return json.loads(docker("exec", container, "python", "-c", script, timeout=60).stdout)


def container_memory(container: str) -> dict[str, float | None]:
    script = "for f in memory.current memory.peak; do test -f /sys/fs/cgroup/$f && cat /sys/fs/cgroup/$f || echo 0; done"
    try:
        lines = docker("exec", container, "sh", "-c", script, timeout=30).stdout.splitlines()
        current, peak = (int(lines[index]) if index < len(lines) and lines[index].isdigit() else 0 for index in range(2))
        return {
            "current_mib": round(current / (1024 * 1024), 1) if current else None,
            "peak_mib": round(peak / (1024 * 1024), 1) if peak else None,
        }
    except (subprocess.SubprocessError, ValueError):
        return {"current_mib": None, "peak_mib": None}


def container_exists(name: str) -> bool:
    return bool(docker("ps", "-aq", "--filter", f"name=^{name}$", timeout=30).stdout.strip())


def container_running(name: str) -> bool:
    return bool(docker("ps", "-q", "--filter", f"name=^{name}$", timeout=30).stdout.strip())


def remove_container(name: str) -> None:
    if container_exists(name):
        docker("rm", "-f", name, check=False, timeout=90)


def wait_for_gpu_settle(gpu_index: int, maximum_mib: int, timeout: int = 90) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    last = gpu_snapshot(gpu_index)
    while last is not None and last["used_mib"] > maximum_mib and time.monotonic() < deadline:
        time.sleep(2)
        last = gpu_snapshot(gpu_index)
    return last


def start_container(args: argparse.Namespace, image: str) -> float:
    remove_container(args.container)
    reference_mount = f"{args.reference_audio.resolve()}:{CONTAINER_REFERENCE_PATH}:ro"
    run_args = [
        "run",
        "-d",
        "--init",
        "--name",
        args.container,
        "-p",
        f"{args.port}:8808",
        "--gpus",
        "all",
        "-e",
        f"CUDA_VISIBLE_DEVICES={args.gpu_index}",
        "-e",
        "VOXCPM_DEVICE=auto",
        "-e",
        f"VOXCPM_BACKEND={args.active_backend}",
        "-e",
        "VOXCPM_MODEL_ID=openbmb/VoxCPM2",
        "-e",
        "VOXCPM_LOAD_DENOISER=0",
        "-e",
        "VOXCPM_LOAD_ASR=0",
        "-e",
        "VOXCPM_OPTIMIZE=0",
        "-e",
        "VOXCPM_LOCAL_FILES_ONLY=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "VOXCPM_NANO_INFERENCE_TIMESTEPS=10",
        "-e",
        "VOXCPM_NANO_MAX_NUM_BATCHED_TOKENS=4096",
        "-e",
        "VOXCPM_NANO_MAX_NUM_SEQS=1",
        "-e",
        "VOXCPM_NANO_MAX_MODEL_LEN=4096",
        "-e",
        "VOXCPM_NANO_GPU_MEMORY_UTILIZATION=0.49",
        "-e",
        "VOXCPM_NANO_ENFORCE_EAGER=0",
        "-e",
        "TORCHINDUCTOR_COMPILE_THREADS=1",
        "-v",
        f"{args.cache_volume}:/app/.cache/huggingface",
        "-v",
        reference_mount,
        image,
    ]
    docker(*run_args, timeout=90)
    return wait_ready(args.base_url)


def run_call(
    base_url: str,
    payload: dict[str, Any],
    gpu_index: int,
    sample_interval: float,
    baseline_mib: int | None,
) -> dict[str, Any]:
    with GpuSampler(gpu_index, sample_interval) as sampler:
        result = generate_audio(base_url, payload)
    result["gpu"] = sampler.summary(baseline_mib)
    return result


def run_backend(
    args: argparse.Namespace,
    backend: str,
    image: str,
    scenarios: list[Scenario],
) -> dict[str, Any]:
    print(f"\n[{backend}] starting {image}", flush=True)
    args.active_backend = backend
    api_ready_seconds = start_container(args, image)
    try:
        initial_status = request_json("GET", f"{args.base_url}/tts/status")
        actual_backend = initial_status.get("backend")
        if backend == "nano" and actual_backend != "nano":
            raise RuntimeError(f"Expected backend {backend!r}, container reported {actual_backend!r}")
        if backend == "native" and actual_backend not in (None, "native"):
            raise RuntimeError(f"Expected legacy native backend, container reported {actual_backend!r}")
        baseline = gpu_snapshot(args.gpu_index)
        baseline_mib = int(baseline["used_mib"]) if baseline is not None else None
        memory_before = container_memory(args.container)

        print(f"[{backend}] cold model load", flush=True)
        cold = run_call(args.base_url, scenarios[0].payload, args.gpu_index, args.sample_interval, baseline_mib)
        memory_after_load = container_memory(args.container)

        scenario_results = []
        total = len(scenarios) * (args.warmup_calls + args.calls)
        progress = 0
        for scenario in scenarios:
            print(f"[{backend}] {scenario.code} {scenario.label}", flush=True)
            warmups = []
            for warmup_index in range(args.warmup_calls):
                progress += 1
                result = run_call(args.base_url, scenario.payload, args.gpu_index, args.sample_interval, baseline_mib)
                warmups.append(result)
                print(
                    f"  [{progress:02d}/{total:02d}] warmup {warmup_index + 1}/{args.warmup_calls}: "
                    f"{result['wall_seconds']:.3f}s, RTF {result['rtf']:.3f}",
                    flush=True,
                )
            measurements = []
            for call_index in range(args.calls):
                progress += 1
                result = run_call(args.base_url, scenario.payload, args.gpu_index, args.sample_interval, baseline_mib)
                measurements.append(result)
                print(
                    f"  [{progress:02d}/{total:02d}] call {call_index + 1}/{args.calls}: "
                    f"{result['wall_seconds']:.3f}s, {result['audio_seconds']:.3f}s audio, RTF {result['rtf']:.3f}",
                    flush=True,
                )
            scenario_results.append(
                {
                    "code": scenario.code,
                    "label": scenario.label,
                    "description": scenario.description,
                    "payload": scenario.payload,
                    "warmups": warmups,
                    "measurements": measurements,
                    "summary": summarize_measurements(measurements),
                    "container_memory_after": container_memory(args.container),
                }
            )

        final_status = request_json("GET", f"{args.base_url}/tts/status")
        final_gpu = gpu_snapshot(args.gpu_index)
        return {
            "backend": backend,
            "image": image_metadata(image),
            "runtime_versions": runtime_versions(args.container),
            "api_ready_seconds": round(api_ready_seconds, 3),
            "status_before_load": initial_status,
            "status_after": final_status,
            "gpu_baseline": baseline,
            "cold_request": cold,
            "gpu_final": final_gpu,
            "gpu_final_delta_mib": (
                int(final_gpu["used_mib"]) - baseline_mib
                if final_gpu is not None and baseline_mib is not None
                else None
            ),
            "container_memory_before_load": memory_before,
            "container_memory_after_load": memory_after_load,
            "container_memory_final": container_memory(args.container),
            "scenarios": scenario_results,
        }
    except Exception:
        logs = docker("logs", "--tail", "120", args.container, check=False, timeout=30).stdout
        if logs:
            print(f"\n[{backend}] container logs:\n{logs}", flush=True)
        raise
    finally:
        remove_container(args.container)


def git_metadata() -> dict[str, Any]:
    revision = command(["git", "rev-parse", "HEAD"], timeout=30).stdout.strip()
    status = command(["git", "status", "--short"], timeout=30).stdout
    diff = command(["git", "diff", "--binary", "HEAD"], timeout=30).stdout.encode("utf-8")
    return {
        "revision": revision,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def append_results(path: Path, run: dict[str, Any]) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": 1, "runs": []}
    data.setdefault("runs", []).append(run)
    data["latest"] = run
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        temporary = Path(file.name)
    temporary.replace(path)


def value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(path: Path, run: dict[str, Any]) -> None:
    by_backend = {item["backend"]: item for item in run["backends"]}
    native = by_backend.get("native")
    nano = by_backend.get("nano")
    lines = [
        "# Backend Comparison",
        "",
        f"Latest official run: `{run['started_at']}`",
        "",
        f"- Hardware: `{run['hardware'].get('name', 'unknown')}` with `{run['hardware'].get('total_mib', 'unknown')} MiB` VRAM; driver `{run['hardware'].get('driver', 'unknown')}`",
        f"- Model: `openbmb/VoxCPM2`; steps: `{run['settings']['inference_timesteps']}`; calls: `{run['settings']['calls']}` after `{run['settings']['warmup_calls']}` warmup(s) per scenario",
        f"- Revision: `{run['git']['revision']}`; dirty worktree: `{str(run['git']['dirty']).lower()}`",
        f"- Comment: {run.get('comment') or 'None'}",
        "",
    ]
    if native and nano:
        lines.extend(
            [
                "## Warmed Results",
                "",
                "Lower latency and RTF are better. Speedup is native median RTF divided by Nano median RTF.",
                "",
                "| Scenario | Native median | Nano median | Speedup | Native RTF | Nano RTF | Native p95 RTF | Nano p95 RTF |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        native_scenarios = {item["code"]: item for item in native["scenarios"]}
        nano_scenarios = {item["code"]: item for item in nano["scenarios"]}
        speedups = []
        for scenario in run["scenarios"]:
            n = native_scenarios[scenario["code"]]["summary"]
            v = nano_scenarios[scenario["code"]]["summary"]
            speedup = n["rtf_median"] / v["rtf_median"] if v["rtf_median"] else None
            if speedup is not None:
                speedups.append(speedup)
            lines.append(
                f"| {scenario['code']} - {scenario['label']} | {n['wall_median_seconds']:.3f}s | "
                f"{v['wall_median_seconds']:.3f}s | {value(speedup, 2)}x | {n['rtf_median']:.3f} | "
                f"{v['rtf_median']:.3f} | {n['rtf_p95']:.3f} | {v['rtf_p95']:.3f} |"
            )
        geometric_speedup = math.exp(statistics.fmean(math.log(item) for item in speedups)) if speedups else None
        lines.extend(["", f"Geometric mean Nano speedup across scenarios: **{value(geometric_speedup, 2)}x**.", ""])

    lines.extend(
        [
            "## Startup And Memory",
            "",
            "VRAM is whole-device use. Delta values subtract each backend's pre-load baseline.",
            "",
            "| Backend | API ready | Cold request | Cold RTF | Cold VRAM delta | Final VRAM delta | Container RAM current | Container RAM peak |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for backend in run["backends"]:
        cold = backend["cold_request"]
        memory = backend["container_memory_final"]
        lines.append(
            f"| {backend['backend']} | {backend['api_ready_seconds']:.3f}s | {cold['wall_seconds']:.3f}s | "
            f"{cold['rtf']:.3f} | {value(cold['gpu']['peak_delta_mib'], 0)} MiB | "
            f"{value(backend['gpu_final_delta_mib'], 0)} MiB | {value(memory['current_mib'], 1)} MiB | "
            f"{value(memory['peak_mib'], 1)} MiB |"
        )

    lines.extend(["", "## Provenance", ""])
    for backend in run["backends"]:
        versions = backend["runtime_versions"]
        lines.append(
            f"- `{backend['backend']}`: image `{backend['image']['tag']}` / `{backend['image']['id']}`; "
            f"Python `{versions.get('python')}`, PyTorch `{versions.get('torch')}`, "
            f"Transformers `{versions.get('transformers')}`, Nano-vLLM-VoxCPM `{versions.get('nano-vllm-voxcpm')}`."
        )
    lines.extend(
        [
            "",
            "See [DETAILS.md](DETAILS.md) for methodology and [runs.json](runs.json) for every request measurement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VoxCPM2 native PyTorch and Nano-vLLM Docker backends.")
    parser.add_argument("--native-image", default="voxcpmtts:benchmark-native")
    parser.add_argument("--nano-image", default="voxcpmtts:benchmark-nano")
    parser.add_argument("--container", default="voxcpmtts_benchmark")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cache-volume", default="voxcpmtts_hf_cache")
    parser.add_argument("--reference-audio", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--limit-scenarios", type=int, default=0)
    parser.add_argument("--backends", nargs="+", choices=("native", "nano"), default=("native", "nano"))
    parser.add_argument("--stop-container", action="append", default=[])
    parser.add_argument("--comment", default="")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.calls < 1 or args.warmup_calls < 0:
        parser.error("--calls must be at least 1 and --warmup-calls must not be negative")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be greater than zero")
    if args.limit_scenarios < 0:
        parser.error("--limit-scenarios must not be negative")
    args.base_url = f"http://127.0.0.1:{args.port}"
    return args


def main() -> int:
    args = parse_args()
    if not args.reference_audio.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {args.reference_audio}")
    hardware = gpu_snapshot(args.gpu_index)
    if hardware is None:
        raise RuntimeError(f"Unable to inspect NVIDIA GPU index {args.gpu_index}")
    docker("volume", "inspect", args.cache_volume, timeout=30)
    images = {"native": args.native_image, "nano": args.nano_image}
    for backend in args.backends:
        image_metadata(images[backend])

    scenarios = build_scenarios(CONTAINER_REFERENCE_PATH)
    if args.limit_scenarios:
        scenarios = scenarios[: args.limit_scenarios]

    previously_running = []
    for name in dict.fromkeys(args.stop_container):
        if name != args.container and container_running(name):
            print(f"Stopping existing VoxCPM container {name} for isolated measurements", flush=True)
            docker("stop", "--time", "60", name, timeout=90)
            previously_running.append(name)

    backends = []
    started_at = now_iso()
    try:
        baseline_target = int((gpu_snapshot(args.gpu_index) or hardware)["used_mib"]) + 128
        for backend in args.backends:
            wait_for_gpu_settle(args.gpu_index, baseline_target)
            backends.append(run_backend(args, backend, images[backend], scenarios))
    finally:
        remove_container(args.container)
        for name in previously_running:
            print(f"Restoring previously running container {name}", flush=True)
            docker("start", name, check=False, timeout=90)

    run = {
        "schema": 1,
        "started_at": started_at,
        "finished_at": now_iso(),
        "comment": args.comment,
        "git": git_metadata(),
        "hardware": hardware,
        "settings": {
            "model_id": "openbmb/VoxCPM2",
            "gpu_index": args.gpu_index,
            "calls": args.calls,
            "warmup_calls": args.warmup_calls,
            "sample_interval_seconds": args.sample_interval,
            "inference_timesteps": 10,
            "cfg_value": 2.0,
            "output_format": "wav",
            "concurrency": 1,
            "nano_gpu_memory_utilization": 0.49,
            "nano_enforce_eager": False,
        },
        "scenarios": [
            {"code": item.code, "label": item.label, "description": item.description} for item in scenarios
        ],
        "backends": backends,
    }
    if args.no_write:
        print("\nSmoke run completed; results were not written.", flush=True)
    else:
        append_results(args.results, run)
        write_summary(args.summary, run)
        print(f"\nRecorded raw results in {args.results}", flush=True)
        print(f"Updated comparison report at {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
