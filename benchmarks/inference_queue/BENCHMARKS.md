# Inference Queue

## FIFO Implementation A/B

The same image and concurrent four-client workload were run with the original semaphore path (`VOXCPM_NANO_QUEUE_SIZE=0`), then the bounded FIFO (`8`), then the semaphore control again. Each run used five rounds and 20 stable-seed calls.

| Implementation | Concurrent wall | Throughput vs control mean | WAV matches | Peak active | Peak queue |
|---|---:|---:|---:|---:|---:|
| Semaphore control A | 41.32s | - | 20/20 | n/a | 0 |
| Bounded FIFO | 41.96s | +0.14% | 20/20 | 1 | 3 |
| Semaphore control B | 42.73s | - | 20/20 | n/a | 0 |

The semaphore-control mean was 42.02s. A 0.14% FIFO throughput difference is measurement noise: the scheduler is performance-neutral. Its benefits are bounded admission, FIFO fairness, queue telemetry, request progress, and deterministic HTTP 429 overload handling rather than faster inference.

## Backlog Behavior

Sequential submission leaves no request waiting; concurrent submission releases all clients together. Both modes use the same single GPU worker, payloads, stable seeds, and warmed model. This measures the value of keeping prepared work available, which applies to both the semaphore and FIFO implementations.

| Run | Candidate | Clients | Rounds | Sequential wall | Queued wall | Throughput change | WAV matches | Peak queue |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20261022-080908 | Rejected candidate: Bounded FIFO scheduler validation at concurrency one | 4 | 5 | 70.87s | 50.54s | +40.3% | 20/20 | 3 |
| 20261022-083122 | Rejected candidate: Optimized bounded FIFO scheduler at concurrency one | 4 | 5 | 65.15s | 47.35s | +37.6% | 20/20 | 3 |
| 20261022-085508 | Accepted candidate: Pure-ASGI bounded FIFO scheduler at concurrency one | 4 | 5 | 59.12s | 42.55s | +38.9% | 20/20 | 3 |

Within the accepted FIFO service, concurrent submission changed aggregate request throughput by **+38.9%** relative to one-at-a-time HTTP submission. All **20** paired fixed-seed WAV files were byte-identical. This gain must not be attributed to the FIFO itself; the queue-disabled controls showed the same behavior.

| Mode | Calls | Median batch | Median latency | p95 latency | Requests/s | Audio speed | Peak active | Peak queued |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential | 20 | 11.640s | 2.932s | 3.543s | 0.338 | 3.48x | 1 | 0 |
| queued | 20 | 8.319s | 6.476s | 8.528s | 0.470 | 4.83x | 1 | 3 |

Higher queued-request latency is expected because it includes time waiting for the single GPU worker. The relevant gain is aggregate batch wall time and throughput, while output hashes guard deterministic quality.

The accepted candidate also passed the complete release baseline as run `20261022-085849`: all 312 WAV hashes and ASR results matched accepted run `20261022-054748`, total TTS time changed by +0.21%, and no errors occurred.

See [DETAILS.md](DETAILS.md) for controls and interpretation.
