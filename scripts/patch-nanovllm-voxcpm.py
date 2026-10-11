"""Apply the VoxCPMTTS runtime patch to the pinned Nano-vLLM-VoxCPM wheel."""

from __future__ import annotations

import site
from importlib.metadata import version
from pathlib import Path


EXPECTED_VERSION = "2.0.4"
PACKAGE = "nano-vllm-voxcpm"

ORIGINAL_ENCODE_LATENTS = '''    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        wav = wav.to(torch.float32).cuda()
        return (
            self.vae.encode(wav, self.vae.sample_rate)
            .permute(0, 2, 1)
            .view(-1, self.feat_dim)
            .to(torch.float32)
            .cpu()
            .numpy()
        )
'''

PATCHED_ENCODE_LATENTS = '''    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        wav = wav.to(torch.float32).cuda()
        try:
            return (
                self.vae.encode(wav, self.vae.sample_rate)
                .permute(0, 2, 1)
                .view(-1, self.feat_dim)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
        finally:
            del wav
            torch.cuda.empty_cache()
'''

ORIGINAL_MEMORY_ERROR = '''                    "Please lower 'max_model_len', 'max_num_batched_tokens', 'max_num_seqs', or "
                    "'gpu_memory_utilization'. Advanced users may set "
'''

PATCHED_MEMORY_ERROR = '''                    "Please lower 'max_model_len', 'max_num_batched_tokens', or 'max_num_seqs', "
                    "or increase 'gpu_memory_utilization' when free GPU memory is available. "
                    "Advanced users may set "
'''


def package_root() -> Path:
    for site_root in map(Path, site.getsitepackages()):
        candidate = site_root / "nanovllm_voxcpm"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Nano-vLLM-VoxCPM package was not found")


def replace_once(source: str, original: str, patched: str, label: str) -> tuple[str, bool]:
    if patched in source:
        return source, False
    if source.count(original) != 1:
        raise RuntimeError(f"Expected {label} implementation was not found exactly once")
    return source.replace(original, patched, 1), True


def main() -> None:
    installed_version = version(PACKAGE)
    if installed_version != EXPECTED_VERSION:
        raise RuntimeError(
            f"Refusing to patch {PACKAGE} {installed_version}; expected {EXPECTED_VERSION}"
        )

    root = package_root()
    runner = root / "models" / "voxcpm2" / "runner.py"
    engine = root / "engine" / "model_runner.py"

    runner_source = runner.read_text(encoding="utf-8")
    runner_source, allocator_cleanup_changed = replace_once(
        runner_source,
        ORIGINAL_ENCODE_LATENTS,
        PATCHED_ENCODE_LATENTS,
        "encode_latents",
    )
    engine_source = engine.read_text(encoding="utf-8")
    engine_source, memory_error_changed = replace_once(
        engine_source,
        ORIGINAL_MEMORY_ERROR,
        PATCHED_MEMORY_ERROR,
        "KV-cache memory error",
    )
    if not allocator_cleanup_changed and not memory_error_changed:
        print(f"{root} is already patched")
        return

    if allocator_cleanup_changed:
        runner.write_text(runner_source, encoding="utf-8")
    if memory_error_changed:
        engine.write_text(engine_source, encoding="utf-8")
    print(f"Patched Nano-vLLM runtime behavior in {root}")


if __name__ == "__main__":
    main()
