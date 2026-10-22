# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import asyncio
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
NANO_BADCASE_RATIO_THRESHOLD = 4.5
NANO_GENERATION_LIMIT_RATIO = 6.0
NANO_GPU_MEMORY_TARGET_BYTES = 8 * 1024**3
NANO_GPU_MEMORY_MAX_UTILIZATION = 0.90


class _ConcurrentNanoServer:
    """Drive Nano's async pool from concurrent synchronous request threads."""

    def __init__(self, server: object) -> None:
        self._server = server
        self._loop = server.loop
        self._pool = server.server_pool
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="voxcpm-nano-event-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Nano-vLLM event loop did not start")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def _result(self, awaitable: object) -> object:
        return asyncio.run_coroutine_threadsafe(awaitable, self._loop).result()

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        return self._result(self._pool.encode_latents(wav, wav_format))

    def generate(self, **kwargs: object) -> Generator[object, None, None]:
        async_generator = self._pool.generate(**kwargs)
        try:
            while True:
                try:
                    yield self._result(async_generator.__anext__())
                except StopAsyncIteration:
                    return
        finally:
            self._result(async_generator.aclose())

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._result(self._pool.stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
            if not self._thread.is_alive():
                self._loop.close()


def _bounded_generation_length(
    text: str,
    tokenizer: Callable[[str], Sequence[int]],
    requested_max: int,
    ratio_threshold: float,
) -> tuple[int, float]:
    if requested_max < 1:
        raise ValueError("max_len must be greater than zero")
    if ratio_threshold <= 0:
        raise ValueError("retry_badcase_ratio_threshold must be greater than zero")
    token_count = max(1, len(tokenizer(text)))
    badcase_threshold = token_count * ratio_threshold
    generation_limit = min(requested_max, int(token_count * NANO_GENERATION_LIMIT_RATIO + 10))
    return generation_limit, badcase_threshold


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


def _nano_gpu_memory_utilization(device_index: int) -> float:
    raw = os.getenv("VOXCPM_NANO_GPU_MEMORY_UTILIZATION", "auto").strip().lower()
    if raw == "auto":
        total_memory = int(torch.cuda.get_device_properties(device_index).total_memory)
        if total_memory <= 0:
            raise RuntimeError("CUDA reported an invalid total memory size")
        return min(NANO_GPU_MEMORY_MAX_UTILIZATION, NANO_GPU_MEMORY_TARGET_BYTES / total_memory)

    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise ValueError("VOXCPM_NANO_GPU_MEMORY_UTILIZATION must be 'auto' or in the range (0, 1]")
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
        gpu_memory_utilization = _nano_gpu_memory_utilization(device_index)
        if os.getenv("VOXCPM_NANO_GPU_MEMORY_UTILIZATION", "auto").strip().lower() == "auto":
            total_gib = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
            print(
                "VoxCPMTTS selected Nano-vLLM GPU memory utilization "
                f"{gpu_memory_utilization:.3f} for a {total_gib:.1f} GiB GPU",
                flush=True,
            )
        self.inference_timesteps = _env_int("VOXCPM_NANO_INFERENCE_TIMESTEPS", 10)
        max_num_seqs = _env_int("VOXCPM_NANO_MAX_NUM_SEQS", 1)
        self._generation_lock = threading.BoundedSemaphore(max_num_seqs)
        raw_server = NanoEngine.from_pretrained(
            model=model_path,
            devices=[device_index],
            inference_timesteps=self.inference_timesteps,
            max_num_batched_tokens=_env_int("VOXCPM_NANO_MAX_NUM_BATCHED_TOKENS", 4096),
            max_num_seqs=max_num_seqs,
            max_model_len=_env_int("VOXCPM_NANO_MAX_MODEL_LEN", 4096),
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=_env_bool("VOXCPM_NANO_ENFORCE_EAGER", False),
        )
        model_info = raw_server.get_model_info()
        self._server = _ConcurrentNanoServer(raw_server) if max_num_seqs > 1 else raw_server
        self.tts_model = SimpleNamespace(sample_rate=int(model_info["sample_rate"]))
        self._text_tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(model_path))
        self._generation_state = threading.local()
        self._normalizer_lock = threading.Lock()
        self._denoiser_lock = threading.Lock()
        self.last_successful_seed = None
        self.text_normalizer = None
        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path:
            self.enable_denoiser(zipenhancer_model_path)

    @property
    def last_successful_seed(self) -> int | None:
        state = getattr(self, "_generation_state", None)
        return getattr(state, "last_successful_seed", None) if state is not None else None

    @last_successful_seed.setter
    def last_successful_seed(self, value: int | None) -> None:
        if not hasattr(self, "_generation_state"):
            self._generation_state = threading.local()
        self._generation_state.last_successful_seed = value

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openbmb/VoxCPM2",
        load_denoiser: bool = False,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
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
                revision=revision,
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
        retry_badcase_ratio_threshold: float = NANO_BADCASE_RATIO_THRESHOLD,
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
            with self._normalizer_lock:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer

                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)

        generation_limit, badcase_threshold = _bounded_generation_length(
            text,
            self._text_tokenizer,
            max_len,
            retry_badcase_ratio_threshold,
        )
        if retry_badcase_max_times < 1:
            raise ValueError("retry_badcase_max_times must be greater than zero")

        temporary_paths: list[str] = []
        actual_prompt_path = prompt_wav_path
        actual_reference_path = reference_wav_path
        try:
            if denoise and self.denoiser is not None:
                with self._denoiser_lock:
                    if prompt_wav_path:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary_file:
                            temporary_paths.append(temporary_file.name)
                        self.denoiser.enhance(
                            prompt_wav_path,
                            output_path=temporary_paths[-1],
                            normalize_loudness=False,
                        )
                        actual_prompt_path = temporary_paths[-1]
                    if reference_wav_path:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temporary_file:
                            temporary_paths.append(temporary_file.name)
                        self.denoiser.enhance(
                            reference_wav_path,
                            output_path=temporary_paths[-1],
                            normalize_loudness=False,
                        )
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
                    is_badcase = len(chunks) >= badcase_threshold
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
