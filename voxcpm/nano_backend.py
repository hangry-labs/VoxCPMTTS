# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import os
import re
import secrets
import tempfile
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Generator, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download


MAX_GENERATION_SEED = 0xFFFFFFFF


def _bounded_generation_length(
    text: str,
    tokenizer: Callable[[str], Sequence[int]],
    requested_max: int,
    ratio_threshold: float,
    patch_size: int,
) -> tuple[int, float]:
    if requested_max < 1:
        raise ValueError("max_len must be greater than zero")
    if ratio_threshold <= 0:
        raise ValueError("retry_badcase_ratio_threshold must be greater than zero")
    if patch_size < 1:
        raise ValueError("patch_size must be greater than zero")
    token_count = max(1, len(tokenizer(text)))
    badcase_threshold = token_count * ratio_threshold
    latent_limit = min(requested_max, int(badcase_threshold + 10))
    return max(1, (latent_limit + patch_size - 1) // patch_size), badcase_threshold


def _materialize_seed(seed: int | None) -> int:
    value = secrets.randbelow(MAX_GENERATION_SEED + 1) if seed is None else int(seed)
    if not 0 <= value <= MAX_GENERATION_SEED:
        raise ValueError("seed must be between 0 and 4294967295")
    return value


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _env_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in the range (0, 1]")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


class NanoVoxCPM:
    """Compatibility adapter between the Hangry Labs API and Nano-vLLM VoxCPM."""

    backend_name = "nano"

    def __init__(
        self,
        model_path: str,
        zipenhancer_model_path: str | None = None,
        enable_denoiser: bool = False,
        device: str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Nano-vLLM requires an NVIDIA CUDA GPU")
        if not device.startswith("cuda"):
            raise ValueError("Nano-vLLM supports CUDA devices only")

        from nanovllm_voxcpm import VoxCPM as NanoEngine
        from transformers import LlamaTokenizerFast

        from .model.utils import mask_multichar_chinese_tokens

        device_index = int(device.split(":", 1)[1]) if ":" in device else 0
        self.inference_timesteps = _env_int("VOXCPM_NANO_INFERENCE_TIMESTEPS", 10)
        self._generation_lock = threading.Lock()
        self._server = NanoEngine.from_pretrained(
            model=model_path,
            devices=[device_index],
            inference_timesteps=self.inference_timesteps,
            max_num_batched_tokens=_env_int("VOXCPM_NANO_MAX_NUM_BATCHED_TOKENS", 4096),
            max_num_seqs=_env_int("VOXCPM_NANO_MAX_NUM_SEQS", 1),
            max_model_len=_env_int("VOXCPM_NANO_MAX_MODEL_LEN", 4096),
            gpu_memory_utilization=_env_float("VOXCPM_NANO_GPU_MEMORY_UTILIZATION", 0.49),
            enforce_eager=_env_bool("VOXCPM_NANO_ENFORCE_EAGER", False),
        )
        model_info = self._server.get_model_info()
        self.tts_model = SimpleNamespace(sample_rate=int(model_info["sample_rate"]))
        self._patch_size = int(model_info["patch_size"])
        self._text_tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(model_path))
        self.last_successful_seed: int | None = None
        self.text_normalizer = None
        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path:
            self.enable_denoiser(zipenhancer_model_path)

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openbmb/VoxCPM2",
        load_denoiser: bool = False,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        device: str = "cuda:0",
        **_: object,
    ) -> "NanoVoxCPM":
        if os.path.isdir(hf_model_id):
            model_path = hf_model_id
        else:
            model_path = snapshot_download(
                repo_id=hf_model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        return cls(
            model_path=model_path,
            zipenhancer_model_path=zipenhancer_model_id,
            enable_denoiser=load_denoiser,
            device=device,
        )

    def enable_denoiser(self, zipenhancer_model_path: str) -> None:
        if self.denoiser is not None:
            return
        from .zipenhancer import ZipEnhancer

        self.denoiser = ZipEnhancer(zipenhancer_model_path)

    def _encode_audio(self, path: str) -> bytes:
        audio_path = Path(path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        audio_format = audio_path.suffix.lstrip(".").lower() or "wav"
        return self._server.encode_latents(audio_path.read_bytes(), audio_format)

    def _generate_chunks(
        self,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        reference_wav_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_len: int = 2000,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        seed: int | None = None,
        **_: object,
    ) -> Generator[np.ndarray, None, None]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if inference_timesteps != self.inference_timesteps:
            raise ValueError(
                "Nano-vLLM fixes inference_timesteps when the engine starts; "
                f"this instance uses {self.inference_timesteps}"
            )
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if normalize:
            if self.text_normalizer is None:
                from .utils.text_normalize import TextNormalizer

                self.text_normalizer = TextNormalizer()
            text = self.text_normalizer.normalize(text)

        generation_limit, badcase_threshold = _bounded_generation_length(
            text,
            self._text_tokenizer,
            max_len,
            retry_badcase_ratio_threshold,
            self._patch_size,
        )
        if retry_badcase_max_times < 1:
            raise ValueError("retry_badcase_max_times must be greater than zero")

        temporary_paths: list[str] = []
        actual_prompt_path = prompt_wav_path
        actual_reference_path = reference_wav_path
        try:
            if denoise and self.denoiser is not None:
                if prompt_wav_path:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary_file:
                        temporary_paths.append(temporary_file.name)
                    self.denoiser.enhance(prompt_wav_path, output_path=temporary_paths[-1])
                    actual_prompt_path = temporary_paths[-1]
                if reference_wav_path:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary_file:
                        temporary_paths.append(temporary_file.name)
                    self.denoiser.enhance(reference_wav_path, output_path=temporary_paths[-1])
                    actual_reference_path = temporary_paths[-1]

            with self._generation_lock:
                prompt_latents = self._encode_audio(actual_prompt_path) if actual_prompt_path else None
                reference_latents = self._encode_audio(actual_reference_path) if actual_reference_path else None
                current_seed = _materialize_seed(seed)
                generation_kwargs = {
                    "target_text": text,
                    "prompt_latents": prompt_latents,
                    "prompt_text": prompt_text or "",
                    "ref_audio_latents": reference_latents,
                    "max_generate_length": generation_limit,
                    "cfg_value": cfg_value,
                }
                if not retry_badcase:
                    for chunk in self._server.generate(**generation_kwargs, seed=current_seed):
                        self.last_successful_seed = current_seed
                        yield np.asarray(chunk, dtype=np.float32).reshape(-1)
                    return

                for attempt in range(retry_badcase_max_times):
                    chunks = [
                        np.asarray(chunk, dtype=np.float32).reshape(-1)
                        for chunk in self._server.generate(**generation_kwargs, seed=current_seed)
                    ]
                    is_badcase = len(chunks) * self._patch_size >= badcase_threshold
                    if is_badcase and attempt + 1 < retry_badcase_max_times:
                        current_seed = (current_seed + 1) & MAX_GENERATION_SEED
                        continue
                    self.last_successful_seed = current_seed
                    yield from chunks
                    break
        finally:
            for temporary_path in temporary_paths:
                Path(temporary_path).unlink(missing_ok=True)

    def generate(self, *args: object, **kwargs: object) -> np.ndarray:
        chunks = list(self._generate_chunks(*args, **kwargs))
        if not chunks:
            raise RuntimeError("Nano-vLLM returned no audio")
        return np.concatenate(chunks)

    def generate_streaming(self, *args: object, **kwargs: object) -> Generator[np.ndarray, None, None]:
        if kwargs.pop("retry_badcase", False):
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
        kwargs["retry_badcase"] = False
        yield from self._generate_chunks(*args, **kwargs)

    def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.stop()
