# Inference Queue Details

This benchmark measures whether a bounded FIFO improves aggregate API
throughput around a single Nano-vLLM GPU worker without changing generated
speech.

## Controls

- The service must report exactly one inference worker.
- Sequential mode submits one request at a time and never builds a backlog.
- Queued mode releases four clients together, allowing one active request and
  three waiting requests.
- Both modes use the same English design prompt, voice direction, 10 inference
  steps, WAV output, and stable 32-bit seed for each paired call.
- Text normalization, loudness normalization, denoising, long-text splitting,
  and randomized seeds are disabled.
- Two warmup calls run before measurement. Round order alternates between
  sequential-first and queued-first to reduce thermal and ordering bias.
- Queue telemetry is sampled from `/tts/status`. The run fails if more than one
  inference is active, no waiting request is observed, any HTTP request fails,
  or any paired WAV differs byte-for-byte.

For an implementation A/B, run the same image twice with
`VOXCPM_NANO_QUEUE_SIZE=0` and `VOXCPM_NANO_QUEUE_SIZE=8`. Pass
`--expect-queue disabled` for the size-zero semaphore control and leave the
default expectation for the FIFO run. This separates the benefit of callers
submitting concurrently from any benefit attributable to the explicit FIFO.

## Interpretation

Queued clients include their wait for the single GPU worker, so their median
request latency is expected to rise. The useful comparison is total batch wall
time, requests per second, and aggregate generated-audio speed. Any gain comes
from overlapping request preparation and completed-audio handling around the
serialized GPU stage; it is not concurrent model inference.

The queue-disabled A/B control is essential for attribution. Two semaphore
control runs completed the concurrent workload in 41.32 s and 42.73 s (42.02 s
mean); the FIFO completed it in 41.96 s, or +0.14% throughput versus the
control mean. All 20 cross-mode WAV hashes matched. The explicit FIFO is
therefore performance-neutral. Concurrent submission itself produced the
large throughput improvement, and that behavior already existed around the
original semaphore.

The benchmark is deliberately smaller than the multilingual release baseline.
Use `task benchmark-baseline` separately to detect broad speed, memory, or ASR
quality regressions against established releases.

## Release Regression Check

The accepted queue candidate was also run through the complete 312-call,
24-language release baseline as run `20261022-085849`, compared with accepted
Nano run `20261022-054748`.

| Check | Result |
|---|---:|
| WAV SHA-256 matches | 312/312 |
| Audio-duration matches | 312/312 |
| Normalized ASR transcript matches | 312/312 |
| ASR similarity-score matches | 312/312 |
| Total TTS time | 462.414s vs 461.444s (+0.21%) |
| Median RTF | 0.2717 vs 0.2707 |
| p95 RTF | 0.3276 vs 0.3294 |
| Peak VRAM delta | 7911 MiB vs 8330 MiB |
| Errors | 0 |

The reference-conditioning cache recorded one miss followed by 78 hits and
held one 31,744-byte entry at the end of the run. An earlier implementation
using FastAPI function middleware was rejected after it increased total TTS
time by about 13%; pure ASGI request tracking restored baseline performance by
forwarding streaming response bodies directly.

Raw run history is stored in the git-ignored
`benchmark-artifacts/inference-queue/runs.json`.
