<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=en">
    <img src="https://raw.githubusercontent.com/hangry-labs/VoxCPMTTS/main/assets/voxcpmtts_logo_horizontal.webp" alt="Hangry Labs VoxCPMTTS logo" width="900">
  </a>
</p>

<p align="center"><a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=en">English</a> · <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=nb">Norsk bokmål</a> · <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=pl">Polski</a> · <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=ja">日本語</a> · <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=zh">简体中文</a> · <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=es">Español</a></p>

<p align="center"><a href="https://hangry-labs.github.io/VoxCPMTTS/examples/?lang=en"><strong>🎧 Listen to voice examples</strong></a></p>

# Hangry Labs VoxCPMTTS

Local VoxCPM2 speech generation, voice design and reference cloning in one Docker
container, with Nano-vLLM inference, a browser Studio, streaming audio,
OpenAI-compatible speech and optional MCP tools.

The commands below run the published **v1.1** release, pinned to matching verified
Docker Hub and GHCR OCI indexes.

## Quick Start

With Docker, NVIDIA GPU support and a CUDA 12.8-compatible driver installed, run:

**Full image**, with the pinned VoxCPM2 and multilingual Whisper Base assets for
offline use after pulling:

```bash
docker run -d --name voxcpmtts --restart unless-stopped -p 127.0.0.1:8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:v1.1@sha256:cc5a1f186cd8baeed3fe777b94ebe8a22648dbd30d5100a498bad46b860210a9
```

Open **[http://localhost:8808](http://localhost:8808)**. Interactive API docs are at
**[http://localhost:8808/tts/docs](http://localhost:8808/tts/docs)**.

For a smaller initial download, choose the **tiny image**:

```bash
docker run -d --name voxcpmtts --restart unless-stopped -p 127.0.0.1:8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:v1.1_tiny@sha256:ca0b5ca41289a931238410d2fcb7c7b8462ed71d4e0a69d8cb6b8cbb17ac2fe6
```

Tiny downloads the models into the same volume on first use. Keep it online until
the required models are cached. Choose either full or tiny, and keep one VoxCPMTTS
model-serving container per GPU. Change `CUDA_VISIBLE_DEVICES=0` to select another
physical GPU.

The `voxcpmtts_data` volume preserves model caches, saved voices, reference audio,
dialogue scripts and settings across container replacement. Full images seed
baked assets into it; tiny images reuse the same cache.

Need a command for another operating system, registry or network-access mode?
Use the [installation command picker](https://hangrylabs.app/software/voxcpmtts/install?os=windows&hardware=nvidia&image=full&registry=dockerhub&access=computer).

## What You Get

- Create voices from natural-language direction, or clone authorized reference recordings
- Choose directed reference cloning or transcript-guided continuation
- Build multi-speaker scenes in Magic, with SSML and SSML-H import/export
- Adjust per-turn direction, language, rate, pitch, volume and loudness leveling
- Generate and lock individual takes, rearrange speech and pauses, and reuse accepted audio
- Save voices and scripts with portraits, tags, search and restorable creation recipes
- Record, upload, trim and transcribe reference audio locally
- Compare original and processed audio, with optional non-destructive finishing
- Generate long-form speech with protection that splits near natural boundaries
- Play progressive MP3 streams, with queue progress, cancellation and timing feedback
- Connect OpenAI-compatible speech, native HTTP APIs or separately enabled MCP tools
- Use the Studio in English, Norwegian Bokmål, Polish, Japanese, Simplified Chinese or Spanish

VoxCPM2 supports [30 languages and nine Chinese dialect groups](https://hangry-labs.github.io/VoxCPMTTS/SUPPORTED_LANGUAGES.html).
Interface language and speech language are separate choices. The
[listening gallery](https://hangry-labs.github.io/VoxCPMTTS/examples/?lang=en) includes voice-variety samples,
translated introductions, cross-language cloning and a featured Magic dialogue.

VoxCPM supports delivery direction with reference cloning. Transcript-guided
continuation uses the reference transcript instead; the Studio keeps these two
conditioning modes separate.

## Browser UI

### Build complete dialogue scripts

Assemble scenes in Magic, assign saved or script-defined voices, set each turn's
delivery, generate and lock takes, and save the complete script as portable SSML-H.

<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=en#build-complete-dialogue-scripts">
    <img src="https://raw.githubusercontent.com/hangry-labs/VoxCPMTTS/main/assets/screenshots/generate-ui.webp" alt="VoxCPMTTS Generate workspace showing a directed three-speaker dialogue and saved scripts" width="1200">
  </a>
</p>

### Design, clone and reuse voices

Create voices from direction or recordings, lock seeds for consistency, attach
portraits and tags, refine the audio, and reuse saved characters across scripts.

<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/?lang=en#design-clone-and-reuse-voices">
    <img src="https://raw.githubusercontent.com/hangry-labs/VoxCPMTTS/main/assets/screenshots/design-ui.webp" alt="VoxCPMTTS Design workspace showing a locked-seed narrator and reusable saved voices" width="1200">
  </a>
</p>

## Image Tags

| Variant | Stable release | Rolling tag | Model assets |
| --- | --- | --- | --- |
| Full | `v1.1` | `latest` | VoxCPM2 and reference-transcription models baked in |
| Tiny | `v1.1_tiny` | `latest_tiny` | Download into the persistent cache on first use |

Both variants are published to `hangrylabs/voxcpmtts` and
`ghcr.io/hangry-labs/voxcpmtts`. The pinned full/tiny commands above use verified
top-level OCI index digests. For GHCR, replace `hangrylabs/voxcpmtts` with
`ghcr.io/hangry-labs/voxcpmtts` and keep the same tag and digest.

Rolling tags deliberately follow `main` and remain unpinned. Fixed releases use
`vX.Y` or `vX.Y.Z`, with the corresponding `_tiny` tag. Historical fixed commands
remain in the [README version history](https://hangry-labs.github.io/VoxCPMTTS/#version-history).

## Persistent Data And Runtime Settings

Reuse `voxcpmtts_data` at `/app/persistent` when replacing the container. Deleting
the volume removes its stored models, voices, recordings, scripts and settings.
The Quick Start binds the service to this computer only; use the command picker
for other access modes. Recording requires localhost or HTTPS and permission.

The runtime uses Python 3.13, CUDA 12.8 and a prebuilt FlashAttention wheel.
Image builds install binary Python wheels; Triton performs normal first-use
kernel compilation inside the container.

| Setting | Default and behavior |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=0` | Select the first physical NVIDIA GPU |
| `VOXCPM_NANO_GPU_MEMORY_UTILIZATION=auto` | Target an approximately 8 GiB engine budget relative to card capacity, capped at 90% |
| `VOXCPM_NANO_MAX_NUM_SEQS=1` | Keep exact fixed-seed repeatability; opt into `2` for higher aggregate throughput with batching-dependent output |
| `VOXCPM_NANO_QUEUE_SIZE=8` | Bounded waiting FIFO at concurrency one; `0` retains the semaphore path |
| `VOXCPM_NANO_LATENT_CACHE_ENTRIES=32` | Bounded CPU cache of exact reference-conditioning bytes; `0` disables it |
| `VOXCPM_NANO_LATENT_CACHE_BYTES=268435456` | Reference-cache limit of 256 MiB; `0` disables it |
| `VOXCPM_NANO_INFERENCE_TIMESTEPS=10` | Diffusion steps fixed at engine startup; requests must use the same value |
| `VOXCPM_NANO_ENFORCE_EAGER=0` | Keep CUDA graphs enabled; set `1` for troubleshooting |
| `TORCHINDUCTOR_COMPILE_THREADS=1` | Avoid retaining unused compiler-worker pools between engine reloads |
| `VOXCPM_LOAD_ASR=1` | Enable on-demand reference transcription |
| `VOXCPM_ASR_DEVICE=cpu` | Keep optional Whisper transcription off the TTS GPU |
| `VOXCPMTTS_API_KEY` | Optional Bearer key for `/v1` routes |

The automatic GPU fraction is about `0.67` on a 12 GiB card and `0.50` on a
16 GiB card. At concurrency one, CPU preparation and final encoding can overlap
around the single GPU worker. Concurrency two can raise aggregate throughput
and individual latency; keep one when exact output repeatability matters.

The UI can transcribe uploaded or recorded reference samples before guided
cloning. Pinned `openai/whisper-base` loads only when requested and defaults to
CPU. Full includes these assets; tiny downloads them into the cache. Set
`VOXCPM_LOAD_ASR=0` to disable transcription, or enter transcripts manually.

<details markdown="1">
<summary><strong>Older v1.0 images on 12 GiB GPUs</strong></summary>

The fixed `v1.0` images use a `0.49` memory fraction. Add
`VOXCPM_NANO_GPU_MEMORY_UTILIZATION=0.67` for a 12 GiB card. Despite the old
Nano-vLLM v2.0.4 error wording, lowering the value after a negative KV-cache block
calculation reduces the available budget further.

</details>

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

Voice cloning accepts a reference path visible inside the container. Place the
reference at `/app/persistent/reference.wav` in the mounted product volume, or
use the upload routes described in the [API guide](https://hangry-labs.github.io/VoxCPMTTS/#api-usage):

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"This voice follows the reference sample.","ref_audio":"/app/persistent/reference.wav","output_format":"mp3"}' \
  -o cloned.mp3
```

Transcript-guided cloning:

```bash
curl -X POST "http://localhost:8808/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"The model continues from the reference voice.","ref_audio":"/app/persistent/reference.wav","ref_text":"Transcript of the reference audio.","output_format":"mp3"}' \
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

## MCP

Enable the compact endpoint in **System > MCP access**, then connect to `http://localhost:8808/mcp/`. It provides five focused tools for health, languages, saved voices, speech inspection, and recommended simple generation. The separately controlled `http://localhost:8808/mcp/advanced/` endpoint adds SSML/SSML-H, voice design, one-off reference cloning, and saved-profile management.

Both endpoints are off by default, have no separate application authentication, and should be enabled only for trusted clients or behind an authenticated reverse proxy. Settings persist in `voxcpmtts_data`.

Generated speech is returned as an expiring download URL rather than raw audio in protocol context. The default TTL is one hour and the accepted range is 60 to 86,400 seconds. Set `VOXCPM_MCP_BASE_URL` to the externally reachable server origin when clients or other containers need to consume those URLs. Advanced local reference files belong in `/app/persistent/mcp-input`; expiring output is stored in `/app/persistent/mcp-output`.

## Measured Runtime

On an RTX 5070 Ti, the controlled 24-language baseline measured Nano-vLLM at
`0.271` median real-time factor versus `1.134` for the native backend, a **4.19x
speedup**. Both used 10 inference steps and concurrency one. Aggregate transcript
fidelity was comparable. See the [benchmark suite](https://hangry-labs.github.io/VoxCPMTTS/benchmarks/BENCHMARKS.html)
for the workload, VRAM measurements, language tables and later paired regressions.

## Responsible Use And Privacy

VoxCPM2 supports highly realistic voice cloning. Do not use this image for unauthorized voice cloning, impersonation, fraud, harassment, scams, or any illegal or unethical activity. Only clone voices when you have the rights and consent to do so.


Generation runs locally. First-use model downloads contact the model host.
Reference transcription uses the local pinned Whisper model when requested.
Saved recordings and scripts remain in the persistent product volume; review
what you share or export. Native routes and MCP require trusted access or
appropriate network controls; the optional Bearer key protects `/v1` only.

## Project Links

- [Project website and documentation](https://hangry-labs.github.io/VoxCPMTTS/)
- [Listening gallery](https://hangry-labs.github.io/VoxCPMTTS/examples/?lang=en)
- [GitHub repository](https://github.com/hangry-labs/VoxCPMTTS)
- [API and integration guide](https://hangry-labs.github.io/VoxCPMTTS/#api-usage)
- [Supported languages](https://hangry-labs.github.io/VoxCPMTTS/SUPPORTED_LANGUAGES.html)
- [Local development](https://hangry-labs.github.io/VoxCPMTTS/#local-development)
- [Docker Hub tags](https://hub.docker.com/r/hangrylabs/voxcpmtts/tags)
- [Upstream VoxCPM project](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2 model](https://huggingface.co/openbmb/VoxCPM2)
- [Hangry Labs](https://hangrylabs.app/)

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
