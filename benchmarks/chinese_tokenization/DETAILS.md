# Chinese Tokenization Qualification

## Scope

This checks the prefixed Chinese token correction associated with
[VoxCPM PR 447](https://github.com/OpenBMB/VoxCPM/pull/447). The pinned VoxCPM2
vocabulary contains 92 SentencePiece-prefixed Chinese multi-character entries
without matching plain entries. Previously, those pieces bypassed character
splitting. The correction normalizes the marker consistently in the local
length-estimation wrapper and Nano's actual inference wrapper.

No model weights, dependencies, or language support were changed.

Qualification completed on 2026-10-29. Recommendation: keep the correction.
The focused Chinese cases improve from zero to five exact ASR matches, while
all 312 standard multilingual WAVs remain byte-identical to the accepted baseline.

## Controls

- RTX 5070 Ti, 16,303 MiB, NVIDIA driver 610.88; one Vox container at a time.
- `openbmb/VoxCPM2` revision `32279effe8c19989596f05d353d1447f51d9e915`.
- Nano-vLLM-VoxCPM 2.0.5, concurrency one, CUDA graphs enabled, 4,096-token
  model/batch limits, GPU-memory utilization 0.49.
- Ten inference steps, CFG 2.0, stable SHA-256-derived seed per call ID,
  `randomize_seed=false`, WAV output at 48 kHz.
- Text normalization, denoising, and loudness normalization disabled. Cloning
  uses the same `examples/original_clone.mp3` reference in both images.
- Existing `benchmarks/run_baseline.py` generation, ASR, scoring, and sampling
  methods. Qwen3-ASR-0.6B-hf receives the explicit language `Chinese`.
- ASR exactness ignores punctuation, spaces, and case. It is a comparative
  intelligibility check, not a native-speaker listening score.

The [focused manifest](manifest.json) contains three affected phrases, with
spaces before them to exercise prefixed tokenization: company introduction,
personal introduction, and screening time. Two design texts have two seeded
repeats each; the reference-clone text has one. Five calls per image are a
targeted regression check, not a broad estimate of Chinese speech quality.

## Image Provenance

| Role | Local image | Immutable image/index ID |
|---|---|---|
| Before | `voxcpmtts:pre-tokenizer447` | `sha256:f79e5c1ce90af1a7b273cb9e064014d2aa87be19ee1fbce786482be8d1975755` |
| Corrected | `voxcpmtts:tokenizer447` | `sha256:6ae61ee1c9da66e4992d571fbb01b31499322535db23060ab3b032eccf1a78d2` |

The corrected image overlays the current application and guarded Nano patch
onto the preserved dependency/model image using the existing benchmark
Dockerfile. No dependencies were downloaded or compiled. This is not a fresh
dependency build; inherited VERSION/build metadata and OCI source labels describe
the original base rather than a newly published release.
Other source changes concern optional text cleanup and CLI batch exit status;
neither is exercised by these normalization-disabled API calls.

## Focused Results

Runs: before `20261029-065844`; corrected `20261029-070003`.

| Metric | Before | Corrected |
|---|---:|---:|
| Completed / errors | 5 / 0 | 5 / 0 |
| Exact ASR matches | 0/5 | 5/5 |
| Mean transcript similarity | 79.86% | 100.00% |
| Minimum similarity | 70.59% | 100.00% |
| Total generated audio | 25.92 s | 23.68 s |
| Total measured generation time | 7.170 s | 6.707 s |
| Median RTF | 0.2748 | 0.2828 |
| p95 RTF | 0.3079 | 0.3102 |
| Peak sampled VRAM delta | 7,829 MiB | 7,872 MiB |
| Container RAM peak (cgroup total) | 14,986.6 MiB | 8,444.7 MiB |

The output durations change, so lower total generation time is not evidence of
a speedup. This is a correctness change with a demonstrated benefit on the
selected phrases, not a general Chinese quality or performance claim.

| Call | Before transcript | Corrected transcript | Similarity before / after |
|---|---|---|---:|
| Design 1, repeat 1 | 这是Office ChatT。我们是一家制作本地语音工具的公司。 | 这是公司简介。我们是一家制作本地语音工具的公司。 | 70.59% / 100% |
| Design 1, repeat 2 | 这是以巴查赛。我们是一家制作本地语音工具的公司。 | 这是公司简介。我们是一家制作本地语音工具的公司。 | 81.82% / 100% |
| Design 2, repeat 1 | 接下来是成绩。我喜欢阅读，也喜欢探索新的技术。 | 接下来是个人简介。我喜欢阅读，也喜欢探索新的技术。 | 85.71% / 100% |
| Design 2, repeat 2 | 接下来是Chip Yo。我喜欢阅读，也喜欢探索新的技术。 | 接下来是个人简介。我喜欢阅读，也喜欢探索新的技术。 | 78.26% / 100% |
| Reference clone | 下面是《商福克》这部电影，将在下周五与大家见面。 | 下面是上映时间：这部电影将在下周五与大家见面。 | 82.93% / 100% |

## Multilingual Regression

Corrected run `20261029-070100` covers the existing 24-language, 312-call
fixed-seed benchmark. The output-regression control is accepted run
`20261022-085849`, not either rejected October 22 outlier. Runtime package
versions match, including Python 3.13.17, PyTorch 2.8.0+cu128, Transformers
5.19.0, Nano 2.0.5, and Triton 3.4.0.

| Metric | Accepted baseline | Corrected |
|---|---:|---:|
| Completed / TTS errors / ASR errors | 312 / 0 / 0 | 312 / 0 / 0 |
| Total generated audio | 1,730.640 s | 1,730.640 s |
| Total measured generation time | 462.414 s | 449.375 s |
| Median RTF | 0.2717 | 0.2623 |
| p95 RTF | 0.3276 | 0.3166 |
| Realtime speed | 3.74x | 3.85x |
| Exact ASR matches | 42.63% | 42.63% |
| Mean transcript similarity | 95.92% | 95.92% |
| Minimum similarity | 23.66% | 23.66% |
| Peak sampled VRAM delta | 7,911 MiB | 7,873 MiB |
| Container RAM at completion | 7,283.5 MiB | 8,262.7 MiB |
| Container RAM peak (cgroup total) | 7,540.3 MiB | 8,518.1 MiB |

All 312 WAV SHA-256 hashes, durations, normalized ASR transcripts, exactness
flags, and similarity scores are unchanged. The standard Chinese subset also
retains its previous 99.56% mean similarity. The focused fixture adds coverage
that the standard texts missed.

Measured generation time is 2.82% lower than the accepted fixed-seed baseline,
and 3.92% lower than historical Nano run `20261007-184048` (467.698 s).
These single-session timing differences are not evidence of a tokenizer-driven
speedup. The historical run used different seed conditions, so it is performance
context, not the waveform-equivalence control. There is no observed throughput
or quality regression in this qualification.

Host-RAM accounting is not equivalent between sessions: the full-run cgroup
peak is 977.8 MiB higher than the accepted baseline, while the short pre-fix
control peaks much higher than either corrected run. These counters include
file cache as well as application memory, and this method does not separate
them. Cache/page-accounting differences are a possible explanation, not a
confirmed attribution. Do not interpret these totals as proof of either a RAM
saving or a tokenizer-caused RAM regression; that would require a separately
controlled memory breakdown. The measured GPU footprint remains comparable.

Detailed standard reports: [speed](../speed/BENCHMARKS.md),
[GPU memory](../memory/gpu/BENCHMARKS.md), and
[speech quality](../speech-quality/BENCHMARKS.md).

## Additional Checks

- Built-candidate CPU regression suite: **264 passed, two skipped**.
- Real pinned-tokenizer probe: 284 inputs agree with the intended policy;
  local/Nano token streams match, with no unknown IDs introduced.
- Corrected `/tts/stream` design and reference-clone requests both returned
  progressive MP3 at 48 kHz and exact normalized Chinese ASR transcripts.
  Decoding the responses with the image's FFmpeg confirmed nonempty playable
  audio. This is a content/transport smoke check, not a streaming-latency
  benchmark; first-use reference encoding was not warmed beforehand.

## Reproduction

Prepare the corrected image from the preserved local baseline:

```powershell
docker build --file benchmarks/backend_comparison/Dockerfile.nano --build-arg BASE_IMAGE=voxcpmtts:pre-tokenizer447 --tag voxcpmtts:tokenizer447 .
```

Run the focused fixture once per image, replacing the ASR URL with an available
Qwen3-ASR service. `--no-write` prevents this short check from replacing the
standard multilingual reports; it still prints each generation and ASR result.

```powershell
python benchmarks/run_baseline.py --backends nano --nano-image voxcpmtts:pre-tokenizer447 --manifest benchmarks/chinese_tokenization/manifest.json --languages Chinese --design-sentences 2 --repeats 2 --clone-repeats 1 --asr-url http://ASR_HOST:PORT --no-write
python benchmarks/run_baseline.py --backends nano --nano-image voxcpmtts:tokenizer447 --manifest benchmarks/chinese_tokenization/manifest.json --languages Chinese --design-sentences 2 --repeats 2 --clone-repeats 1 --asr-url http://ASR_HOST:PORT --no-write
```

Run the standard full regression, writing the normal suite reports:

```powershell
python benchmarks/run_baseline.py --backends nano --nano-image voxcpmtts:tokenizer447 --asr-url http://ASR_HOST:PORT
```

Raw histories and retained WAVs are local, git-ignored benchmark artifacts.
Focused histories and corrected audio are under
`benchmark-artifacts/chinese-tokenization/`; pre-fix non-exact clips are under
`benchmark-artifacts/speech-quality/20261029-065844/nano/`.
