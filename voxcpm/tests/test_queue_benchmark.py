from __future__ import annotations

import io
import wave

from benchmarks.inference_queue.run_benchmark import percentile, stable_seed, summarize, wav_duration


def wav_bytes(*, frames: int = 24_000, sample_rate: int = 24_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\0\0" * frames)
    return output.getvalue()


def test_queue_benchmark_helpers_are_deterministic() -> None:
    assert stable_seed("call-1") == stable_seed("call-1")
    assert stable_seed("call-1") != stable_seed("call-2")
    assert wav_duration(wav_bytes()) == 1.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 2.9


def test_queue_benchmark_summary_uses_batch_throughput() -> None:
    batches = [
        {
            "batch_wall_seconds": 2.0,
            "queue": {"peak_active": 1, "peak_queued": 3},
            "calls": [
                {"wall_seconds": 1.0, "audio_seconds": 4.0},
                {"wall_seconds": 2.0, "audio_seconds": 4.0},
            ],
        }
    ]

    result = summarize("queued", batches)

    assert result["calls"] == 2
    assert result["requests_per_second"] == 1.0
    assert result["aggregate_realtime_speed"] == 4.0
    assert result["peak_active"] == 1
    assert result["peak_queued"] == 3
