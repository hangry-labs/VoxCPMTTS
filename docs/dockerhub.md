<p align="center">
  <a href="https://hangrylabs.app/">
    <img src="https://github.com/Hangry-Labs/VoxCPMTTS/raw/main/assets/voxcpmtts_logo_horizontal.webp" alt="Hangry Labs VoxCPMTTS logo">
  </a>
</p>

# Hangry Labs VoxCPMTTS

Easy-to-run VoxCPM2 text-to-speech Docker images with Nano-vLLM inference, a browser UI, and an HTTP API included.

This Hangry Labs fork is built for people who want realistic multilingual text to speech, voice design, and voice cloning without a long setup. Install Docker, run one command, open the local UI, or call the API from your own application.

## Responsible Use

VoxCPM2 supports highly realistic voice cloning. Do not use this image for unauthorized voice cloning, impersonation, fraud, harassment, scams, or any illegal or unethical activity. Only clone voices when you have the rights and consent to do so.

## Project Links

- GitHub repository: https://github.com/Hangry-Labs/VoxCPMTTS
- Project page: https://hangry-labs.github.io/VoxCPMTTS/examples/
- Upstream VoxCPM project: https://github.com/OpenBMB/VoxCPM
- Upstream model: https://huggingface.co/openbmb/VoxCPM2
- Hangry Labs: https://hangrylabs.app/

## Quick Start

Run with NVIDIA GPU support:

```bash
docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 hangrylabs/voxcpmtts:latest
```

Run on another physical GPU, for example index `1`:

```bash
docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=1 hangrylabs/voxcpmtts:latest
```

Then open:

http://localhost:8808

The standard full image (`latest` or a fixed `vX.Y` release) includes the VoxCPM2 model assets for offline use after the image is pulled.

The runtime uses Nano-vLLM on Python 3.13 with CUDA 12.8 and a prebuilt FlashAttention wheel. Triton performs normal first-use kernel JIT compilation inside the container. Image builds install binary Python wheels only.

The default `VOXCPM_NANO_GPU_MEMORY_UTILIZATION=0.49` gives the engine a budget of about 7.5 GiB on a 16 GiB GPU. Run one VoxCPMTTS container per GPU. The service reuses one cached model for `auto` and `cuda:0` and serializes generation with a concurrency of one. CUDA graphs are enabled for Nano-vLLM acceleration; set `VOXCPM_NANO_ENFORCE_EAGER=1` only for troubleshooting. `TORCHINDUCTOR_COMPILE_THREADS=1` prevents unused compile-worker pools from retaining RAM between engine reloads. The image also releases temporary CUDA allocator blocks after reference-audio encoding so different input lengths do not accumulate VRAM high-water allocations.

Nano-vLLM fixes the diffusion step count at engine startup. The image default is 10; set `VOXCPM_NANO_INFERENCE_TIMESTEPS` before startup to change it, and use the same value in API requests.

The moving tiny tag is `latest_tiny`; fixed releases use the `vX.Y_tiny` pattern. Tiny images keep runtime dependencies but skip baked model assets, and are intended for persistent-volume workflows where the Hugging Face cache is warmed on first online use:

```bash
docker run --name voxcpmtts-tiny --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v voxcpmtts_hf_cache:/app/.cache/huggingface hangrylabs/voxcpmtts:latest_tiny
```

## What You Get

- Offline browser workspace for voice design, controllable cloning, transcript-guided cloning, recording, upload, and waveform trimming
- HTTP API for applications and automation
- VoxCPM2 multilingual generation across 30 officially supported languages
- 48 kHz output when using the VoxCPM2 AudioVAE V2 model
- WAV, MP3, FLAC, and OGG output support
- GPU support when Docker/NVIDIA support is available
- Nano-vLLM inference with a single cached GPU engine
- Offline-friendly usage with the standard full image once it is available locally
- Kokoro-shaped compatibility fields such as `voice`, `use_gpu`, `/tts/voices`, `/tts/speakers`, `/tts/stream-formats`, and `/tts/stream`

## API Example

Default API behavior returns WAV:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Hangry Labs VoxCPMTTS","language":"English"}' \
  -o hello.wav
```

Request MP3 when you want compact output:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Hangry Labs VoxCPMTTS","language":"English","output_format":"mp3"}' \
  -o hello.mp3
```

Voice design:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a custom designed voice.","control":"young female, warm, gentle, slightly smiling","output_format":"mp3"}' \
  -o designed.mp3
```

Voice cloning can be called with a reference audio path that is visible inside the container:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"This voice follows the reference sample.","ref_audio":"/data/ref.wav","output_format":"mp3"}' \
  -o cloned.mp3
```

Transcript-guided cloning:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"The model continues from the reference voice.","ref_audio":"/data/ref.wav","ref_text":"Transcript of the reference audio.","output_format":"mp3"}' \
  -o ultimate.mp3
```

Health check:

```bash
curl http://localhost:8808/tts/ping
```

API docs are available at:

http://localhost:8808/tts/docs

## Image Tags

- `latest` and `latest_tiny` are rolling development snapshots from `main`.
- Immutable releases use `vX.Y` / `vX.Y.Z` and matching `_tiny` tags.
- Release documentation pins Docker Hub's top-level OCI digest as `vX.Y@sha256:...` for reproducible deployment.
- Every published tag is mirrored to `ghcr.io/hangry-labs/voxcpmtts`.

Snapshot tags remain intentionally unpinned because they move with `main`. No immutable `v0.1` image has been published yet.

## Measured Runtime

The controlled 24-language baseline measured Nano-vLLM at `0.271` median real-time factor versus `1.134` for the native backend, a **4.19x speedup**, with comparable aggregate transcript fidelity. The complete methodology, raw results, VRAM measurements, and language tables are available in the [benchmark suite](https://github.com/Hangry-Labs/VoxCPMTTS/tree/main/benchmarks).

## Attribution

This is an independently maintained Hangry Labs packaging and serving fork of the original VoxCPM project by OpenBMB, ModelBest, THUHCSI, and contributors:

https://github.com/OpenBMB/VoxCPM

The repository source and upstream VoxCPM source are provided under Apache-2.0.
Original VoxCPM copyright remains with the upstream authors; Hangry Labs
maintains the Docker packaging, Web UI/API integration, documentation, release
tooling, and related modifications in this fork.

The image also contains model assets, Python and Debian packages, FFmpeg, and
NVIDIA CUDA libraries under their respective licenses. The complete image is
not licensed solely under Apache-2.0. Attribution, third-party terms, source
availability, and inspection instructions are documented in the repository:

- https://github.com/Hangry-Labs/VoxCPMTTS/blob/main/NOTICE
- https://github.com/Hangry-Labs/VoxCPMTTS/blob/main/THIRD_PARTY_NOTICES.md
