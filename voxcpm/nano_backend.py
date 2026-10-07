# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Generator

import numpy as np
import torch
from huggingface_hub import snapshot_download

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
                for chunk in self._server.generate(
                    target_text=text,
                    prompt_latents=prompt_latents,
                    prompt_text=prompt_text or "",
                    ref_audio_latents=reference_latents,
                    max_generate_length=max_len,
                    cfg_value=cfg_value,
                ):
                    yield np.asarray(chunk, dtype=np.float32).reshape(-1)
        finally:
            for temporary_path in temporary_paths:
                Path(temporary_path).unlink(missing_ok=True)

    def generate(self, *args: object, **kwargs: object) -> np.ndarray:
        chunks = list(self._generate_chunks(*args, **kwargs))
        if not chunks:
            raise RuntimeError("Nano-vLLM returned no audio")
        return np.concatenate(chunks)

    def generate_streaming(self, *args: object, **kwargs: object) -> Generator[np.ndarray, None, None]:
        yield from self._generate_chunks(*args, **kwargs)

    def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.stop()
