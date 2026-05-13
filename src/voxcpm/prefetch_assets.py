from __future__ import annotations

import os

from huggingface_hub import snapshot_download


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
        asr_id = os.getenv("VOXCPM_ASR_MODEL_ID", "iic/SenseVoiceSmall")
        print(f"Prefetching ModelScope ASR: {asr_id}")
        _prefetch_modelscope(asr_id)


if __name__ == "__main__":
    main()
