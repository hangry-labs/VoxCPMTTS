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
docker run -d --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:v1.1@sha256:cc5a1f186cd8baeed3fe777b94ebe8a22648dbd30d5100a498bad46b860210a9
```

Run on another physical GPU, for example index `1`:

```bash
docker run -d --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=1 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:v1.1@sha256:cc5a1f186cd8baeed3fe777b94ebe8a22648dbd30d5100a498bad46b860210a9
```

Then open:

http://localhost:8808

The standard full image (`latest` or a fixed `vX.Y` release) includes the VoxCPM2 and multilingual Whisper Base model assets for offline use after the image is pulled.

The unified `voxcpmtts_data` volume stores model caches, saved voice profiles, reference samples, and future application settings across container replacement. Full images seed baked assets into it; tiny images download into it on first online use.

The runtime uses Nano-vLLM on Python 3.13 with CUDA 12.8 and a prebuilt FlashAttention wheel. Triton performs normal first-use kernel JIT compilation inside the container. Image builds install binary Python wheels only.

The rolling images default `VOXCPM_NANO_GPU_MEMORY_UTILIZATION=auto`, targeting an approximately 8 GiB engine budget: about `0.67` on a 12 GiB GPU and `0.50` on a 16 GiB GPU, capped at `0.90` on smaller cards. Run one VoxCPMTTS container per GPU. The service reuses one cached model for `auto` and `cuda:0` and defaults to concurrency one. Set `VOXCPM_NANO_MAX_NUM_SEQS=2` for higher aggregate API throughput against the same model copy; individual latency rises and fixed-seed output can vary with concurrent batch composition, so keep `1` when exact repeatability matters. At concurrency one, a bounded eight-call FIFO (`VOXCPM_NANO_QUEUE_SIZE`) overlaps CPU preparation and final encoding around the single GPU worker; set the queue size to `0` to retain the original semaphore path for purely sequential workloads. Reference-conditioning latents are cached as exact content-addressed bytes in CPU memory, with defaults of 32 entries and 256 MiB (`VOXCPM_NANO_LATENT_CACHE_ENTRIES`, `VOXCPM_NANO_LATENT_CACHE_BYTES`); set either limit to `0` to disable caching. CUDA graphs are enabled for Nano-vLLM acceleration; set `VOXCPM_NANO_ENFORCE_EAGER=1` only for troubleshooting. `TORCHINDUCTOR_COMPILE_THREADS=1` prevents unused compile-worker pools from retaining RAM between engine reloads. The image also releases temporary CUDA allocator blocks after reference-audio encoding so different reference lengths do not accumulate VRAM high-water allocations.

The fixed `v1.0` images use `0.49`. For a 12 GiB GPU, add `VOXCPM_NANO_GPU_MEMORY_UTILIZATION=0.67` to the container environment. Despite the Nano-vLLM v2.0.4 error wording, lowering the value after a negative KV-cache block calculation reduces its available budget further.

Nano-vLLM fixes the diffusion step count at engine startup. The image default is 10; set `VOXCPM_NANO_INFERENCE_TIMESTEPS` before startup to change it, and use the same value in API requests.

The UI can transcribe uploaded or freshly recorded reference samples before transcript-guided cloning. The pinned `openai/whisper-base` model loads only when requested and uses CPU by default so VoxCPM2 retains the GPU budget. Set `VOXCPM_LOAD_ASR=0` to disable this feature. The full image contains the ASR assets; the tiny image downloads them into its Hugging Face cache on first use.

The moving tiny tag is `latest_tiny`; fixed releases use the `vX.Y_tiny` pattern. Tiny images keep runtime dependencies but skip baked model assets, and are intended for persistent-volume workflows where the Hugging Face cache is warmed on first online use:

```bash
docker run -d --name voxcpmtts-tiny --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:v1.1_tiny@sha256:ca0b5ca41289a931238410d2fcb7c7b8462ed71d4e0a69d8cb6b8cbb17ac2fe6
```

## What You Get

- Offline browser Design workspace for direction-only creation, controllable cloning, transcript-guided cloning, local reference transcription, recording, upload, waveform trimming, restorable design recipes, and reusable saved voices
- OpenAI-compatible speech API plus the complete native API for applications and automation
- Opt-in compact and advanced Streamable HTTP MCP endpoints with expiring audio links
- SSML and SSML-H for multi-speaker discussions, plays, saved clone voices, and designed speakers
- VoxCPM2 multilingual generation across 30 officially supported languages
- 48 kHz output when using the VoxCPM2 AudioVAE V2 model
- WAV, MP3, FLAC, OGG, Opus, AAC, and raw PCM output support
- GPU support when Docker/NVIDIA support is available
- Nano-vLLM inference with a single cached GPU engine
- Offline-friendly usage with the standard full image once it is available locally
- Common TTS compatibility fields such as `voice`, `use_gpu`, `/tts/voices`, `/tts/speakers`, `/tts/stream-formats`, and `/tts/stream`

## MCP

Enable the compact endpoint in **System > MCP access**, then connect to `http://localhost:8808/mcp/`. It provides five focused tools for health, languages, saved voices, speech inspection, and recommended simple generation. The separately controlled `http://localhost:8808/mcp/advanced/` endpoint adds SSML/SSML-H, voice design, one-off reference cloning, and saved-profile management.

Both endpoints are off by default, have no separate application authentication, and should be enabled only for trusted clients or behind an authenticated reverse proxy. Settings persist in `voxcpmtts_data`.

Generated speech is returned as an expiring download URL rather than raw audio in protocol context. The default TTL is one hour and the accepted range is 60 to 86,400 seconds. Set `VOXCPM_MCP_BASE_URL` to the externally reachable server origin when clients or other containers need to consume those URLs. Advanced local reference files belong in `/app/persistent/mcp-input`; expiring output is stored in `/app/persistent/mcp-output`.

## API Example

OpenAI-compatible clients can use `http://localhost:8808/v1` as their base URL. A direct request uses the standard speech payload:

```bash
curl -X POST "http://localhost:8808/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from Hangry Labs VoxCPMTTS.","voice":"alloy","response_format":"mp3"}' \
  -o hello.mp3
```

The compatibility API accepts standard model and voice aliases, `instructions`, `speed`, and `mp3`, `opus`, `aac`, `flac`, `wav`, or raw `pcm` output. Saved VoxCPMTTS voice names also work as `voice`. Discovery routes are `GET /v1/models` and `GET /v1/audio/voices`. Set `VOXCPMTTS_API_KEY` to require bearer authentication on `/v1` routes. Use the native API below for uploads, SSML, SSML-H, and progressive model-chunk streaming.

### Native API

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

Progressive MP3 streaming:

```bash
curl --no-buffer -X POST "http://localhost:8808/tts/stream" \
  -H "Content-Type: application/json" \
  -d '{"text":"This audio is generated and played progressively.","stream_format":"mp3"}' \
  -o streamed.mp3
```

SSML and SSML-H use the same generation and streaming endpoints with an explicit `input_type` of `ssml` or `ssml-h`. Discover supported elements, limits, voice scopes, and processor behavior at `GET /tts/ssml/capabilities`. Saved voice profile names work in `<voice name="...">`, and SSML-H definitions can create consistent speakers for a complete document.

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

The current immutable release references are:

- Full: `hangrylabs/voxcpmtts:v1.1@sha256:cc5a1f186cd8baeed3fe777b94ebe8a22648dbd30d5100a498bad46b860210a9`
- Tiny: `hangrylabs/voxcpmtts:v1.1_tiny@sha256:ca0b5ca41289a931238410d2fcb7c7b8462ed71d4e0a69d8cb6b8cbb17ac2fe6`

Docker Hub and GHCR expose these same verified top-level OCI index digests. Both images were published from the annotated `v1.1` source tag.

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
