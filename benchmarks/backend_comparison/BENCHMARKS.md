# Backend Comparison

Latest official run: `2026-10-07T16:23:32+00:00`

- Hardware: `NVIDIA GeForce RTX 5070 Ti` with `16303 MiB` VRAM; driver `610.88`
- Model: `openbmb/VoxCPM2`; steps: `10`; calls: `10` after `2` warmup(s) per scenario
- Revision: `d9e163bcccc3bbcaf6927b411e3b7854937bb8a4`; dirty worktree: `true`
- Comment: Historical native image 68787abb versus current Nano backend; same GPU, checkpoint cache, API payloads, 10 steps, concurrency 1

## Warmed Results

Lower latency and RTF are better. Speedup is native median RTF divided by Nano median RTF.

| Scenario | Native median | Nano median | Speedup | Native RTF | Nano RTF | Native p95 RTF | Nano p95 RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| PS - plain_short_english | 4.380s | 1.110s | 4.17x | 1.153 | 0.276 | 1.201 | 0.292 |
| DM - designed_medium_english | 11.118s | 2.747s | 4.28x | 1.192 | 0.279 | 1.276 | 0.304 |
| MN - designed_norwegian | 5.784s | 1.513s | 4.20x | 1.165 | 0.277 | 1.245 | 0.348 |
| CR - reference_clone | 5.408s | 1.252s | 4.17x | 1.171 | 0.281 | 1.212 | 0.305 |

Geometric mean Nano speedup across scenarios: **4.21x**.

## Startup And Memory

VRAM is whole-device use. Delta values subtract each backend's pre-load baseline.

| Backend | API ready | Cold request | Cold RTF | Cold VRAM delta | Final VRAM delta | Container RAM current | Container RAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 6.056s | 19.337s | 5.036 | 5771 MiB | 6459 MiB | 3590.1 MiB | 10866.8 MiB |
| nano | 4.037s | 22.418s | 5.389 | 8351 MiB | 7501 MiB | 3067.5 MiB | 3324.3 MiB |

## Provenance

- `native`: image `voxcpmtts:benchmark-native` / `sha256:68787abb3ecca6fa04d579e43f93d0ce30d3e74d89497450896e1d04fe7d1413`; Python `3.13.16`, PyTorch `2.10.0+cu128`, Transformers `5.3.0`, Nano-vLLM-VoxCPM `None`.
- `nano`: image `voxcpmtts:benchmark-nano` / `sha256:3a4e1ccda4af76e389521374637905c6462974480dac20c476da9414255923f1`; Python `3.13.16`, PyTorch `2.8.0+cu128`, Transformers `5.3.0`, Nano-vLLM-VoxCPM `2.0.4`.

See [DETAILS.md](DETAILS.md) for methodology and [runs.json](runs.json) for every request measurement.
