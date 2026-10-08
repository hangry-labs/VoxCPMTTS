# syntax=docker/dockerfile:1

FROM python:3.13-slim AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    MODELSCOPE_CACHE=/app/.cache/modelscope \
    VOXCPM_PREFETCH_DENOISER=1 \
    VOXCPM_PREFETCH_ASR=1 \
    VOXCPM_ASR_MODEL_ID=openai/whisper-base \
    VOXCPM_ASR_MODEL_REVISION=e37978b90ca9030d5170a5c07aadb050351a65bb

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

FROM dependencies AS app-builder

COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md VERSION /app/
COPY voxcpm /app/voxcpm
COPY assets /app/assets

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install -e . --no-deps

FROM dependencies AS asset-builder

COPY voxcpm/prefetch_assets.py /tmp/prefetch_assets.py

RUN --mount=type=cache,id=hangrylabs-huggingface,target=/root/.cache/huggingface,sharing=locked \
    --mount=type=cache,id=hangrylabs-modelscope,target=/root/.cache/modelscope,sharing=locked \
    HF_HOME=/root/.cache/huggingface \
    MODELSCOPE_CACHE=/root/.cache/modelscope \
    python -u /tmp/prefetch_assets.py \
    && mkdir -p /app/.cache/huggingface /app/.cache/modelscope \
    && cp -a /root/.cache/huggingface/. /app/.cache/huggingface/ \
    && cp -a /root/.cache/modelscope/. /app/.cache/modelscope/

FROM python:3.13-slim AS nano-dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    MODELSCOPE_CACHE=/app/.cache/modelscope \
    VOXCPM_PREFETCH_DENOISER=1 \
    VOXCPM_PREFETCH_ASR=1 \
    VOXCPM_ASR_MODEL_ID=openai/whisper-base \
    VOXCPM_ASR_MODEL_REVISION=e37978b90ca9030d5170a5c07aadb050351a65bb

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ARG NANO_VLLM_VERSION=2.0.4
ARG FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
ARG FLASH_ATTN_WHEEL_SHA256=7dd8c64a414130c82d83a472f5498c1b899fba37a30e2a793a7a4dc5dd61a062

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade --only-binary=:all: pip setuptools wheel

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install --only-binary=:all: \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0 triton==3.4.0

COPY requirements.nano.txt /app/requirements.nano.txt

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install --only-binary=:all: -r /app/requirements.nano.txt \
    && python -m pip install --only-binary=:all: --no-deps "${FLASH_ATTN_WHEEL_URL}#sha256=${FLASH_ATTN_WHEEL_SHA256}" \
    && python -m pip install --only-binary=:all: --no-deps --ignore-requires-python "nano-vllm-voxcpm==${NANO_VLLM_VERSION}"

COPY scripts/patch-nanovllm-voxcpm.py /tmp/patch-nanovllm-voxcpm.py

RUN python /tmp/patch-nanovllm-voxcpm.py

FROM nano-dependencies AS nano-app-builder

COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md VERSION /app/
COPY voxcpm /app/voxcpm
COPY assets /app/assets

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install -e . --no-deps

FROM python:3.13-slim AS nano-asset-dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN --mount=type=cache,id=hangrylabs-pip,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade --only-binary=:all: pip setuptools wheel \
    && python -m pip install --only-binary=:all: huggingface-hub==1.7.1

FROM nano-asset-dependencies AS nano-asset-builder

ENV VOXCPM_PREFETCH_DENOISER=0 \
    VOXCPM_PREFETCH_ASR=1 \
    VOXCPM_ASR_MODEL_ID=openai/whisper-base \
    VOXCPM_ASR_MODEL_REVISION=e37978b90ca9030d5170a5c07aadb050351a65bb

COPY voxcpm/prefetch_assets.py /tmp/prefetch_assets.py

RUN --mount=type=cache,id=hangrylabs-huggingface,target=/root/.cache/huggingface,sharing=locked \
    HF_HOME=/root/.cache/huggingface \
    python -u /tmp/prefetch_assets.py \
    && mkdir -p /app/.cache/huggingface \
    && cp -a /root/.cache/huggingface/. /app/.cache/huggingface/

FROM python:3.13-slim AS runtime-base

LABEL org.opencontainers.image.source="https://github.com/hangry-labs/VoxCPMTTS"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/persistent/models/huggingface \
    MODELSCOPE_CACHE=/app/persistent/models/modelscope \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VOXCPM_DEVICE=auto \
    VOXCPM_MODEL_ID=openbmb/VoxCPM2 \
    VOXCPM_LOAD_ASR=1 \
    VOXCPM_ASR_MODEL_ID=openai/whisper-base \
    VOXCPM_ASR_MODEL_REVISION=e37978b90ca9030d5170a5c07aadb050351a65bb \
    VOXCPM_ASR_DEVICE=cpu \
    VOXCPM_LOAD_DENOISER=1 \
    VOXCPM_OPTIMIZE=0 \
    PORT=8808 \
    HOST=0.0.0.0 \
    UVICORN_RELOAD=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && mkdir -p /app/persistent/models/huggingface /app/persistent/models/modelscope /app/persistent/app /app/persistent/voices /tmp/voxcpmtts-ssml \
    && rm -rf /var/lib/apt/lists/*

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}"

ENV VOXCPMTTS_BUILD_DATE="${BUILD_DATE}" \
    VOXCPMTTS_VCS_REF="${VCS_REF}" \
    BUILD_ID="${BUILD_DATE}@${VCS_REF}"

EXPOSE 8808
VOLUME ["/app/persistent"]

CMD ["python", "-u", "-m", "voxcpm.docker_entrypoint"]

FROM runtime-base AS runtime-app

COPY --from=app-builder /usr/local /usr/local
COPY --from=app-builder /app /app

FROM runtime-app AS native-tiny

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

FROM runtime-app AS native-baked

COPY --from=asset-builder /app/.cache/huggingface /app/baked-models/huggingface
COPY --from=asset-builder /app/.cache/modelscope /app/baked-models/modelscope

FROM runtime-base AS nano-runtime-app

ENV VOXCPM_BACKEND=nano \
    VOXCPM_LOAD_DENOISER=0 \
    VOXCPM_NANO_INFERENCE_TIMESTEPS=10 \
    VOXCPM_NANO_MAX_NUM_BATCHED_TOKENS=4096 \
    VOXCPM_NANO_MAX_NUM_SEQS=1 \
    VOXCPM_NANO_MAX_MODEL_LEN=4096 \
    VOXCPM_NANO_GPU_MEMORY_UTILIZATION=0.49 \
    VOXCPM_NANO_ENFORCE_EAGER=0 \
    TORCHINDUCTOR_COMPILE_THREADS=1

COPY --from=nano-app-builder /usr/local /usr/local
COPY --from=nano-app-builder /app /app

FROM nano-runtime-app AS tiny

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

FROM nano-runtime-app AS baked

COPY --from=nano-asset-builder /app/.cache/huggingface /app/baked-models/huggingface
