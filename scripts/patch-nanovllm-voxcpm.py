"""Apply the VoxCPMTTS runtime patch to the pinned Nano-vLLM-VoxCPM wheel."""

from __future__ import annotations

import site
from importlib.metadata import version
from pathlib import Path


EXPECTED_VERSION = "2.0.4"
PACKAGE = "nano-vllm-voxcpm"

ORIGINAL = '''    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
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

PATCHED = '''    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
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


def package_file() -> Path:
    for package_root in map(Path, site.getsitepackages()):
        candidate = package_root / "nanovllm_voxcpm" / "models" / "voxcpm2" / "runner.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Nano-vLLM-VoxCPM runner.py was not found")


def main() -> None:
    installed_version = version(PACKAGE)
    if installed_version != EXPECTED_VERSION:
        raise RuntimeError(
            f"Refusing to patch {PACKAGE} {installed_version}; expected {EXPECTED_VERSION}"
        )

    runner = package_file()
    source = runner.read_text(encoding="utf-8")
    if PATCHED in source:
        print(f"{runner} is already patched")
        return
    if source.count(ORIGINAL) != 1:
        raise RuntimeError(f"Expected encode_latents implementation was not found exactly once in {runner}")

    runner.write_text(source.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    print(f"Patched CUDA allocator cleanup in {runner}")


if __name__ == "__main__":
    main()
