# Backend Comparison Methodology

This benchmark compares the preserved native PyTorch image from before the Nano migration with the current Nano-vLLM execution path while holding the `openbmb/VoxCPM2` checkpoint, GPU, model cache, request order, API generation parameters, and concurrency constant.

## Controls

- Native uses the preserved local `voxcpmtts:license-audit` image (`sha256:68787abb3ecca6fa04d579e43f93d0ce30d3e74d89497450896e1d04fe7d1413`), built before the Nano migration.
- Nano uses the existing local `voxcpmtts:tiny` image with the current application source and verified Nano reference-encoding cache patch layered on top. Preparing it does not install or compile packages.
- Runtime package versions and immutable image IDs are captured in every run. Differences in PyTorch and dependency versions are part of the historical-versus-current comparison and are shown in the report.
- Only one VoxCPMTTS container runs at a time. The benchmark container is removed before the next backend starts.
- Both backends use one visible GPU, concurrency one, WAV output, `cfg_value=2.0`, and 10 inference steps.
- Denoising, ASR, text normalization, and ffmpeg output conversion are disabled.
- The same Hugging Face cache and `examples/original_clone.mp3` reference are mounted into each container.
- A first cold request records model load and backend initialization separately from warmed measurements.
- Every scenario has unmeasured warmup calls followed by measured calls. Results report median and p95 wall time and real-time factor (wall time divided by generated audio duration).
- Whole-device VRAM is sampled with `nvidia-smi`. Reported deltas subtract the idle baseline captured before model loading; the absolute value can include the desktop and unrelated host consumers.
- Container RAM comes from the Linux cgroup `memory.current` and `memory.peak` counters.

## Workload

| Code | Scenario | Purpose |
|---|---|---|
| PS | Plain short English | Low-latency baseline without a voice prompt |
| DM | Designed medium English | Voice-design path with a longer utterance |
| MN | Designed Norwegian | Multilingual voice-design path |
| CR | Reference clone | Reference-audio encoding plus zero-shot cloning |

Generated speech is stochastic and the two engines do not expose a shared deterministic seed through the project API. Timing and memory are therefore compared over repeated fixed requests; this suite does not claim waveform identity or replace listening and speech-quality evaluation.

## Commands

```powershell
task benchmark-build
task benchmark-backends -- --comment "same revision, idle GPU"
```

Use `task benchmark-backends-smoke` to validate orchestration without writing benchmark history. The smoke still performs one cold model load per backend.
