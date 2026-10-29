# Nano-vLLM-VoxCPM 2.0.5 Synchronization

This benchmark reviews Nano-vLLM-VoxCPM 2.0.5 plus the merged upstream
block-cache fix from PR 89. Performance is compared against the established
2.0.4 baseline from run `20261007-184048`. A same-day 2.0.4 run was retained
for paired output-equivalence evidence but excluded from performance
conclusions because its timing and RAM measurements were clear outliers.

## Result

| Signal | Established 2.0.4 | 2.0.5 + PR 89 | Change |
|---|---:|---:|---:|
| Median RTF | 0.2707 | 0.2707 | unchanged |
| p95 RTF | 0.3153 | 0.3294 | +4.5% |
| Total synthesis time | 467.698 s | 461.444 s | -1.3% |
| Aggregate realtime speed | 3.70x | 3.75x | +1.4% |
| Design median RTF | 0.2754 | 0.2769 | +0.5% |
| Clone median RTF | 0.2581 | 0.2510 | -2.8% |
| Clone total synthesis time | 199.014 s | 192.782 s | -3.1% |
| Peak VRAM delta | 8,491 MiB | 8,330 MiB | -161 MiB |
| Final VRAM delta | 7,533 MiB | 7,288 MiB | -245 MiB |
| Errors | 0 | 0 | unchanged |

The correct performance conclusion is parity. Overall median speed is
identical, total synthesis time differs by only 1.3%, and the small clone gain
is balanced by a 4.5% p95 regression. These differences are modest enough to
be normal run-to-run variation rather than evidence of a broad speed change.

Container RAM is not compared against the established run. The application,
dependency set, and image evolved substantially between October 7 and October
22, so its increase from 3,376.5 MiB to 7,870.3 MiB cannot be attributed to
Nano-vLLM 2.0.5. In the same-day pair, 2.0.5 measured 398 MiB more peak VRAM
and 349.3 MiB more peak container RAM than 2.0.4, but the 2.0.4 run also had
outlier timing and RAM behavior. Additional repeated runs are required before
claiming a package-level memory change.

## Output Equivalence

The same-day runs `20261022-052037` and `20261022-054748` used matching stable
seeds for every call. All 312 corresponding WAV files had identical SHA-256
hashes and durations. All 312 ASR similarity and exact-match results were also
identical. The synchronization therefore preserved generated output exactly
under this controlled concurrency-one workload.

The established baseline used unseeded generation, so its aggregate ASR score
must not be compared directly with the seeded October 22 run. The paired
same-day result is the valid quality comparison.

## Decision

Adopt 2.0.5 plus PR 89. It preserves output, maintains established inference
performance, improves worker-failure handling and first-audio internals, and
bounds the prefix-cache hash map for long-running servers. The decision does
not depend on a claimed speedup.

## Controls

- Hardware: NVIDIA GeForce RTX 5070 Ti, driver 610.88.
- Model: pinned `openbmb/VoxCPM2` revision
  `32279effe8c19989596f05d353d1447f51d9e915`.
- Workload: 312 calls across 24 common VoxCPM2 and Qwen3-ASR languages.
- Modes: 240 voice-design calls and 72 reference-clone calls.
- Runtime: one container, one visible GPU, concurrency one, 10 inference
  steps, WAV output, no denoising, no text normalization, and no output
  conversion.
- The October 22 pair used a stable 32-bit seed derived from each call ID with
  randomization disabled.
- The same LAN Qwen3-ASR service judged staged audio after TTS measurements.

## Images And Runs

| Role | Run | Image ID | Package |
|---|---|---|---|
| Performance reference | `20261007-184048` | `sha256:3a4e1ccda4af76e389521374637905c6462974480dac20c476da9414255923f1` | `nano-vllm-voxcpm==2.0.4` |
| Paired output reference | `20261022-052037` | `sha256:896881513b15712f99bc5a38bc15c717816fe3b730a90cd5790e23146c60f91a` | `nano-vllm-voxcpm==2.0.4` |
| Candidate | `20261022-054748` | `sha256:62a9f53e11c5c5f5d40a745eadd2ea653f524538ec806269c0019e5591e210aa` | `nano-vllm-voxcpm==2.0.5` plus PR 89 |

The 2.0.5 wheel was the published pure-Python wheel with SHA-256
`ae37a585df30d6d708d56c224e7cd9050868462d1203a9e2e92432c2d4193174`.
No dependency wheel was compiled. The image retains VoxCPMTTS's guarded
reference-latent allocator cleanup and corrected KV-cache guidance patches.
PR 89 bounds the prefix-cache hash map by evicting stale entries when blocks
are recycled.

## Command

The paired images were run separately with the same command shape:

```powershell
python benchmarks/run_baseline.py `
  --backends nano `
  --nano-image <preserved-image-tag> `
  --container voxcpmtts_benchmark `
  --port 8811 `
  --gpu-index 0 `
  --cache-volume voxcpmtts_hf_cache `
  --asr-url <qwen3-asr-url> `
  --comment "<run description>"
```

If another model service occupies the selected GPU, stop it before the run or
pass `--stop-container <your-container-name>` to pause and restore that service.

Raw measurements are retained locally in the git-ignored canonical baseline
history. This committed report preserves the reviewed comparison, immutable
image IDs, controls, and key metrics without duplicating multi-megabyte JSON
across benchmark suites.
