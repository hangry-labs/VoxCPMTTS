# Inference Speed

Warmed multilingual API generation. Lower real-time factor (RTF) is better.

| Run | Backend | Calls | Audio | TTS time | Median RTF | p95 RTF | Realtime speed |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261007-184048 | native | 312 | 1740.6s | 2000.9s | 1.134 | 1.272 | 0.87x |
| 20261007-184048 | nano | 312 | 1730.4s | 467.7s | 0.271 | 0.315 | 3.70x |

Latest median Nano speedup: **4.19x**.

See [DETAILS.md](DETAILS.md) for per-language and per-mode results.
