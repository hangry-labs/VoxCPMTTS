FROM python:3.13-slim AS dependencies

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

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

FROM dependencies AS app-builder

COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md VERSION /app/
COPY voxcpm /app/voxcpm
COPY hangrylabs /app/hangrylabs

RUN python -m pip install -e . --no-deps

FROM dependencies AS asset-builder

COPY voxcpm/prefetch_assets.py /tmp/prefetch_assets.py

RUN python -u /tmp/prefetch_assets.py

FROM python:3.13-slim AS runtime-base

LABEL org.opencontainers.image.source="https://github.com/hangry-labs/VoxCPMTTS"

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

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}"

ENV VOXCPMTTS_BUILD_DATE="${BUILD_DATE}" \
    VOXCPMTTS_VCS_REF="${VCS_REF}" \
    BUILD_ID="${BUILD_DATE}@${VCS_REF}"

EXPOSE 8808

CMD ["python", "-u", "-m", "voxcpm.app"]

FROM runtime-base AS runtime-app

COPY --from=app-builder /usr/local /usr/local
COPY --from=app-builder /app /app

FROM runtime-app AS tiny

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

FROM runtime-app AS baked

COPY --from=asset-builder /app/.cache /app/.cache
