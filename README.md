<p>
  <a href="https://github.com/Hangry-Labs/VoxCPMTTS">
    <img src="logo.jpg" alt="Hangry Labs VoxCPMTTS logo">
  </a>
</p>

# Hangry Labs VoxCPMTTS

Easy-to-run Docker packaging for VoxCPM2 with Nano-vLLM inference, a browser UI, and an HTTP API included.

This fork keeps the upstream OpenBMB VoxCPM code, license, and attribution intact, then adds Hangry Labs runtime packaging for local use: Docker images, task automation, API routes under `/tts/*`, format conversion, and a browser UI.

## Docker Quick Start

Run with NVIDIA GPU support:

```bash
docker run -p 8808:8808 --gpus all hangrylabs/voxcpmtts:latest
```

Run on a specific GPU:

```bash
docker run -p 8808:8808 --gpus '"device=0"' hangrylabs/voxcpmtts:latest
```

Open:

http://localhost:8808

API docs:

http://localhost:8808/tts/docs

All published tags are mirrored on Docker Hub as `hangrylabs/voxcpmtts` and GHCR as `ghcr.io/hangry-labs/voxcpmtts`. The moving `latest` and `latest_tiny` tags follow `main`; fixed releases use `vX.Y` and `vX.Y_tiny`.

The full image bakes the VoxCPM2 model assets for offline use after the image is pulled. Tiny images warm a persistent Hugging Face cache on first online use.

The runtime uses Nano-vLLM with Python 3.13, CUDA 12.8, and a prebuilt FlashAttention wheel. Triton still performs normal first-use kernel JIT compilation inside the container; no Python packages or wheels are compiled during the image build.

The default `VOXCPM_NANO_GPU_MEMORY_UTILIZATION=0.49` gives the engine a budget of about 7.5 GiB on a 16 GiB GPU. Run one VoxCPMTTS container per GPU. The service canonicalizes `auto` and `cuda:0` to one cached model instance and serializes generation with a concurrency of one. CUDA graphs are enabled for Nano-vLLM acceleration; set `VOXCPM_NANO_ENFORCE_EAGER=1` only for troubleshooting. `TORCHINDUCTOR_COMPILE_THREADS=1` prevents unused compile-worker pools from retaining RAM between engine reloads. The image also releases temporary CUDA allocator blocks after reference-audio encoding so different input lengths do not accumulate VRAM high-water allocations.

Nano-vLLM fixes the diffusion step count when the engine starts. The image default is 10; set `VOXCPM_NANO_INFERENCE_TIMESTEPS` before startup when a different value is required, then send the same `inference_timesteps` value in API requests.

## Local Development

```bash
task --list
task doctor
task compile
task app
```

This fork is validated primarily through Docker. `task image` builds the full baked offline image, and `task imagerun` runs it without external model cache mounts so the image proves its own baked assets. `task image-tiny` and `task imagerun-tiny` are for online first-use cache workflows. All Docker dependency installs require binary wheels.

## HTTP API

Default generation:

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
  -d '{"text":"This is a designed voice.","control":"young female, warm and gentle","output_format":"mp3"}' \
  -o designed.mp3
```

Voice cloning with a container-visible reference file:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"This follows the reference voice.","ref_audio":"/data/ref.wav","output_format":"mp3"}' \
  -o cloned.mp3
```

Transcript-guided cloning:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"The model continues from the reference voice.","ref_audio":"/data/ref.wav","ref_text":"Transcript of the reference audio.","output_format":"mp3"}' \
  -o guided.mp3
```

Health check:

```bash
curl http://localhost:8808/tts/ping
```

## Runtime Features

- Browser UI for voice design, controllable cloning, and transcript-guided cloning
- HTTP API for applications and automation
- VoxCPM2 multilingual generation across the upstream supported languages
- WAV, MP3, FLAC, and OGG output support
- Lazy model loading so `/tts/ping` and `/tts/status` stay lightweight
- Nano-vLLM inference with a single cached GPU engine
- Docker-first full and tiny image workflows

## Responsible Use

VoxCPM2 supports highly realistic voice cloning. Do not use this project for unauthorized voice cloning, impersonation, fraud, harassment, scams, or any illegal or unethical activity. Only clone voices when you have the rights and consent to do so, and clearly mark generated speech where appropriate.

## Upstream Project

VoxCPM is an OpenBMB project released under Apache-2.0.

- Upstream repository: https://github.com/OpenBMB/VoxCPM
- Upstream model: https://huggingface.co/openbmb/VoxCPM2
- Upstream documentation: https://voxcpm.readthedocs.io/
- Nano-vLLM VoxCPM runtime: https://github.com/a710128/nanovllm-voxcpm

This fork preserves the upstream license, public package code, citation, and attribution. Hangry Labs maintains the Docker packaging, Web UI/API integration, task automation, release tooling, and runtime documentation in this repository.

## Citation

```bibtex
@article{voxcpm2_2026,
  title   = {VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning},
  author  = {VoxCPM Team},
  journal = {GitHub},
  year    = {2026},
}

@article{voxcpm2025,
  title   = {VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation
             and True-to-Life Voice Cloning},
  author  = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and
             Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and
             Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2509.24650},
  year    = {2025},
}
```

## License

The VoxCPMTTS source distribution and upstream VoxCPM source are provided under
the [Apache License 2.0](LICENSE). Original VoxCPM copyright remains with
OpenBMB and the upstream contributors; see [NOTICE](NOTICE).

The Docker images are aggregate distributions that also contain model assets,
Python packages, Debian packages such as FFmpeg, and NVIDIA CUDA libraries under
their respective terms. The image as a whole is not licensed solely under
Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the model,
runtime, source-availability, and redistribution details.
