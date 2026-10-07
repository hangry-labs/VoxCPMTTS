# Speech Quality

Qwen3-ASR is a fixed comparative semantic judge, not ground truth or a replacement for listening tests.

| Run | Backend | Calls | Exact | Mean similarity | Minimum | Repeat consistency | Errors | ASR model |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 20261007-184048 | native | 312 | 46.15% | 95.90% | 37.66% | 18.06% | 0 | Qwen/Qwen3-ASR-0.6B-hf |
| 20261007-184048 | nano | 312 | 46.79% | 96.36% | 38.00% | 20.83% | 0 | Qwen/Qwen3-ASR-0.6B-hf |

Non-exact WAV files are retained locally under the git-ignored `benchmark-artifacts/` directory.
