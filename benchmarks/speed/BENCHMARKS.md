# Inference Speed

Warmed multilingual API generation. Lower real-time factor (RTF) is better.

| Run | Backend | Calls | Audio | TTS time | Median RTF | p95 RTF | Realtime speed |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261007-184048 | native | 312 | 1740.6s | 2000.9s | 1.134 | 1.272 | 0.87x |
| 20261007-184048 | nano | 312 | 1730.4s | 467.7s | 0.271 | 0.315 | 3.70x |
| 20261022-052037 | nano | 312 | 1730.6s | 566.8s | 0.326 | 0.443 | 3.05x |
| 20261022-054748 | nano | 312 | 1730.6s | 461.4s | 0.271 | 0.329 | 3.75x |
| 20261022-080929 | nano | 312 | 1730.6s | 522.3s | 0.309 | 0.373 | 3.31x |
| 20261022-085849 | nano | 312 | 1730.6s | 462.4s | 0.272 | 0.328 | 3.74x |

See [DETAILS.md](DETAILS.md) for per-language and per-mode results.
