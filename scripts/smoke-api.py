from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import tempfile
import time
import unicodedata
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "examples" / "original_clone.mp3"
CASES = [
    {
        "name": "natural",
        "text": "The release candidate is ready for a careful listening test.",
        "seed": 101,
    },
    {
        "name": "warm-design",
        "text": "A warm voice can make technical information easier to follow.",
        "control": "warm adult female narrator, clear and friendly",
        "seed": 202,
    },
    {
        "name": "deep-design",
        "text": "Every component passed its focused runtime validation.",
        "control": "deep adult male narrator, smooth and composed",
        "seed": 303,
    },
    {
        "name": "reference-clone",
        "text": "This sentence checks the reusable reference voice workflow.",
        "clone": True,
        "seed": 404,
    },
]


def normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in value if unicodedata.category(character)[0] in {"L", "N"})


def similarity(expected: str, actual: str) -> float:
    return round(100 * difflib.SequenceMatcher(None, normalized(expected), normalized(actual)).ratio(), 2)


def request_json(method: str, url: str, *, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def multipart(fields: dict[str, str], file_field: str, file_path: Path, content_type: str) -> tuple[bytes, str]:
    boundary = f"----VoxCPMSmoke{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), boundary


def generate(base_url: str, case: dict[str, Any], reference: Path) -> tuple[bytes, dict[str, Any]]:
    payload = {
        "text": case["text"],
        "language": "English",
        "device": "cuda:0",
        "output_format": "wav",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": False,
        "denoise": False,
        "seed": case["seed"],
        "randomize_seed": False,
    }
    if case.get("control"):
        payload["control"] = case["control"]

    if case.get("clone"):
        body, boundary = multipart(
            {"payload": json.dumps(payload)},
            "reference_audio",
            reference,
            "audio/mpeg",
        )
        request = urllib.request.Request(
            f"{base_url}/tts/generate-upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
    else:
        request = urllib.request.Request(
            f"{base_url}/tts/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        audio = response.read()
        content_type = response.headers.get("Content-Type", "")
        returned_seed = response.headers.get("X-VoxCPM-Seed")
    elapsed = time.perf_counter() - started
    if not content_type.startswith("audio/wav") or len(audio) < 1_000:
        raise RuntimeError(f"{case['name']}: invalid WAV response ({content_type!r}, {len(audio)} bytes)")
    if returned_seed != str(case["seed"]):
        raise RuntimeError(f"{case['name']}: expected seed {case['seed']}, got {returned_seed!r}")
    with wave.open(io.BytesIO(audio), "rb") as source:
        duration = source.getnframes() / source.getframerate()
        sample_rate = source.getframerate()
    if duration <= 0:
        raise RuntimeError(f"{case['name']}: generated WAV has no audio frames")
    return audio, {
        "seconds": round(elapsed, 3),
        "audio_seconds": round(duration, 3),
        "rtf": round(elapsed / duration, 3),
        "sample_rate": sample_rate,
        "bytes": len(audio),
        "seed": int(returned_seed),
        "sha256": hashlib.sha256(audio).hexdigest(),
    }


def transcribe(asr_url: str, audio_path: Path) -> str:
    body, boundary = multipart(
        {
            "model": "qwen3-asr",
            "response_format": "verbose_json",
            "temperature": "0",
            "language": "English",
        },
        "file",
        audio_path,
        "audio/wav",
    )
    request = urllib.request.Request(
        f"{asr_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return str(json.loads(response.read().decode("utf-8")).get("text") or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a running VoxCPMTTS API without managing containers.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8808")
    parser.add_argument("--asr-url", help="Optional Qwen3-ASR service used to judge generated speech.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, help="Keep generated WAV files in this directory.")
    parser.add_argument("--min-similarity", type=float, default=75.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    asr_url = args.asr_url.rstrip("/") if args.asr_url else None
    if not args.reference.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {args.reference}")
    status = request_json("GET", f"{base_url}/tts/status")
    if asr_url:
        request_json("GET", f"{asr_url}/health/ready")

    with tempfile.TemporaryDirectory(prefix="voxcpm-smoke-") as temporary:
        output_dir = args.output_dir or Path(temporary)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for case in CASES:
            audio, metrics = generate(base_url, case, args.reference)
            audio_path = output_dir / f"{case['name']}.wav"
            audio_path.write_bytes(audio)
            result = {"name": case["name"], "text": case["text"], **metrics}
            if asr_url:
                transcript = transcribe(asr_url, audio_path)
                result["transcript"] = transcript
                result["similarity"] = similarity(case["text"], transcript)
                if result["similarity"] < args.min_similarity:
                    raise RuntimeError(
                        f"{case['name']}: ASR similarity {result['similarity']:.2f}% "
                        f"is below {args.min_similarity:.2f}%"
                    )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "status": "ok",
        "backend": status.get("backend"),
        "model_id": status.get("model_id"),
        "cases": len(results),
        "asr_judged": bool(asr_url),
        "minimum_similarity": min((item["similarity"] for item in results), default=None),
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
