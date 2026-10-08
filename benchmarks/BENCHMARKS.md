# VoxCPMTTS Benchmark Suite

Benchmarks are manual release-engineering tools, not unit tests. They do not run as part of `task test`, image builds, or releases.

## Initial Baseline

Run `20261007-184048` measured 312 calls per backend across 24 common VoxCPM2 and Qwen3-ASR languages on an NVIDIA GeForce RTX 5070 Ti. Each backend used 10 inference steps, concurrency one, two design prompts repeated five times, and one cloning prompt repeated three times per language.

Common languages: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Polish, Portuguese, Russian, Spanish, Swedish, Tagalog (Qwen label: Filipino), Thai, Turkish, and Vietnamese. Burmese, Hebrew, Khmer, Lao, Norwegian, and Swahili are present in the VoxCPM examples but excluded because the Qwen service does not advertise them.

| Signal | Native | Nano | Result |
|---|---:|---:|---:|
| Median RTF | 1.134 | 0.271 | Nano 4.19x faster |
| p95 RTF | 1.272 | 0.315 | Nano 4.03x faster |
| Mean transcript similarity | 95.90% | 96.36% | Nano +0.46 pp |
| Exact transcript rate | 46.15% | 46.79% | Nano +0.64 pp |
| Peak VRAM delta | 7,849 MiB | 8,491 MiB | Nano +642 MiB |
| Peak container RAM | 10,836.0 MiB | 3,376.5 MiB | Nano -7,459.5 MiB |

The result supports Nano as the default backend: it provides the expected fourfold speedup without an aggregate transcript-fidelity regression. It uses somewhat more peak VRAM under this configuration but substantially less peak system RAM. Generation is stochastic and the API does not expose a shared deterministic seed, so repeated samples characterize each backend's distribution rather than paired identical outputs. Speech recognition is a consistent comparative judge rather than ground truth; listening tests remain necessary for naturalness, speaker similarity, and prosody.

## Suites

| Area | Official task | Isolation | Primary signal |
|---|---|---|---|
| [Inference speed](speed/BENCHMARKS.md) | `task benchmark-baseline` | One VoxCPMTTS backend at a time | Warmed latency and real-time factor by mode and language |
| [GPU memory](memory/gpu/BENCHMARKS.md) | `task benchmark-baseline` | One VoxCPMTTS backend at a time; remote ASR judge | Whole-device VRAM delta and container RAM |
| [Speech quality](speech-quality/BENCHMARKS.md) | `task benchmark-baseline` | Staged TTS followed by Qwen3-ASR judging | Exact transcript rate, normalized similarity, and repeat consistency |
| [Backend pilot](backend_comparison/BENCHMARKS.md) | `task benchmark-backends` | One VoxCPMTTS container at a time | Compact four-scenario diagnostic comparison |
| [SSML dialogue](ssml/BENCHMARKS.md) | `task benchmark-ssml` | Paired staged SSML and SSML-H generation followed by Qwen3-ASR | Ten turn endings, final-tail completion, similarity, latency, and RTF |

Prepare the preserved historical native image and current Nano image before an official comparison:

```powershell
task benchmark-build
task benchmark-baseline -- --comment "description of the change"
```

The baseline computes the intersection of VoxCPM and Qwen3-ASR languages, preserves raw measurements in each suite's `runs.json`, and regenerates summary and detail tables. The ASR service may run on another trusted machine so it does not consume the measured GPU. Do not compare runs collected with different model assets, generation settings, hardware, language sets, or concurrent GPU workloads.
