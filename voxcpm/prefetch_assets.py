from __future__ import annotations

import os

from huggingface_hub import snapshot_download


ASR_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "LICENSE*",
    "README.md",
]


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _prefetch_modelscope(model_id: str) -> None:
    from modelscope import snapshot_download as modelscope_snapshot_download

    modelscope_snapshot_download(model_id)


def main() -> None:
    model_id = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
    print(f"Prefetching VoxCPM model: {model_id}")
    snapshot_download(repo_id=model_id)

    if _enabled(os.getenv("VOXCPM_PREFETCH_DENOISER"), default=True):
        denoiser_id = os.getenv("ZIPENHANCER_MODEL_ID", "iic/speech_zipenhancer_ans_multiloss_16k_base")
        print(f"Prefetching ModelScope denoiser: {denoiser_id}")
        _prefetch_modelscope(denoiser_id)

    if _enabled(os.getenv("VOXCPM_PREFETCH_ASR"), default=True):
        asr_id = os.getenv("VOXCPM_ASR_MODEL_ID", "openai/whisper-base")
        asr_revision = os.getenv(
            "VOXCPM_ASR_MODEL_REVISION",
            "e37978b90ca9030d5170a5c07aadb050351a65bb",
        )
        print(f"Prefetching Hugging Face ASR: {asr_id}@{asr_revision}")
        snapshot_download(
            repo_id=asr_id,
            revision=asr_revision,
            allow_patterns=ASR_ALLOW_PATTERNS,
        )


if __name__ == "__main__":
    main()
