<p>
  <a href="https://github.com/Hangry-Labs/VoxCPMTTS">
    <img src="https://github.com/Hangry-Labs/VoxCPMTTS/raw/main/logo.jpg" alt="Hangry Labs VoxCPMTTS logo">
  </a>
</p>

# Hangry Labs VoxCPMTTS

Easy-to-run VoxCPM2 text-to-speech Docker images with a browser UI and HTTP API included.

This Hangry Labs fork is built for people who want realistic multilingual text to speech, voice design, and voice cloning without a long setup. Install Docker, run one command, open the local UI, or call the API from your own application.

## Responsible Use

VoxCPM2 supports highly realistic voice cloning. Do not use this image for unauthorized voice cloning, impersonation, fraud, harassment, scams, or any illegal or unethical activity. Only clone voices when you have the rights and consent to do so.

## Project Links

- GitHub repository: https://github.com/Hangry-Labs/VoxCPMTTS
- Project page: https://hangry-labs.github.io/VoxCPMTTS/examples/
- Upstream VoxCPM project: https://github.com/OpenBMB/VoxCPM
- Upstream model: https://huggingface.co/openbmb/VoxCPM2
- Hangry Labs: https://nuggies.website/

## Quick Start

Run with NVIDIA GPU support:

```bash
docker run -p 8808:8808 --gpus all hangrylabs/voxcpmtts:v0.1
```

Run on CPU:

```bash
docker run -p 8808:8808 -e VOXCPM_DEVICE=cpu hangrylabs/voxcpmtts:v0.1
```

Run on a specific GPU:

```bash
docker run -p 8808:8808 --gpus "device=1" -e CUDA_VISIBLE_DEVICES=1 hangrylabs/voxcpmtts:v0.1
```

Then open:

http://localhost:8808

The standard `vX.Y` image is the full baked image with VoxCPM2 model assets plus the denoiser and ASR support assets included for offline-friendly use after the image is pulled.

The runtime defaults to `VOXCPM_OPTIMIZE=0` so the slim image does not need a C compiler for first-run Triton compilation. Set `VOXCPM_OPTIMIZE=1` only when you want to test compiled inference.

Tiny tags use the `vX.Y_tiny` pattern. They keep runtime dependencies but skip baked Hugging Face model assets, and are intended for persistent-volume workflows where the cache is warmed on first online use:

```bash
docker run -p 8808:8808 --gpus all -v voxcpmtts_hf_cache:/app/.cache/huggingface hangrylabs/voxcpmtts:v0.1_tiny
```

## What You Get

- Browser UI for voice design, controllable cloning, and transcript-guided cloning
- HTTP API for applications and automation
- VoxCPM2 multilingual generation across 30 officially supported languages
- 48 kHz output when using the VoxCPM2 AudioVAE V2 model
- WAV, MP3, FLAC, and OGG output support
- GPU support when Docker/NVIDIA support is available
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

- Current release tag: `v0.1`
- Future release tags use the same pattern: `vX.Y`
- Tiny tags use the pattern `vX.Y_tiny`

## Attribution

This is an independently maintained Hangry Labs packaging and serving fork of the original VoxCPM project by OpenBMB, ModelBest, THUHCSI, and contributors:

https://github.com/OpenBMB/VoxCPM

License and attribution are preserved in the repository. Original VoxCPM copyright remains with the upstream authors; Hangry Labs maintains the Docker packaging, Web UI/API integration, documentation, release tooling, and related modifications in this fork.
