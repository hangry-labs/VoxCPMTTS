FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    HF_HOME=/app/.cache/huggingface \
    MODELSCOPE_CACHE=/app/.cache/modelscope \
    VOXCPM_PREFETCH_DENOISER=1 \
    VOXCPM_PREFETCH_ASR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE VERSION requirements.txt /app/
COPY voxcpm /app/voxcpm
COPY hangrylabs /app/hangrylabs

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt \
    && python -m pip install -e . --no-deps

FROM base AS baked-builder

RUN python -u -m voxcpm.prefetch_assets

FROM python:3.13-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    MODELSCOPE_CACHE=/app/.cache/modelscope \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VOXCPM_DEVICE=auto \
    VOXCPM_MODEL_ID=openbmb/VoxCPM2 \
    VOXCPM_LOAD_DENOISER=1 \
    VOXCPM_OPTIMIZE=0 \
    PORT=8808 \
    HOST=0.0.0.0 \
    UVICORN_RELOAD=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8808

CMD ["python", "-u", "-m", "voxcpm.app"]

FROM runtime-base AS tiny

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

COPY --from=base /usr/local /usr/local
COPY --from=base /app /app

FROM runtime-base AS baked

COPY --from=baked-builder /usr/local /usr/local
COPY --from=baked-builder /app /app
