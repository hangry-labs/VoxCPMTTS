"""Apply the VoxCPMTTS runtime patch to the pinned Nano-vLLM-VoxCPM wheel."""

from __future__ import annotations

import site
from importlib.metadata import version
from pathlib import Path


EXPECTED_VERSION = "2.0.5"
PACKAGE = "nano-vllm-voxcpm"

ORIGINAL_CHINESE_TOKEN_LOOKUP = r'''    multichar_tokens = {
        token for token in tokenizer.vocab.keys() if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }
'''

PATCHED_CHINESE_TOKEN_LOOKUP = r'''    multichar_tokens = {
        clean
        for token in tokenizer.vocab.keys()
        if len(clean := token.replace("\u2581", "")) >= 2
        and all("\u4e00" <= char <= "\u9fff" for char in clean)
    }
'''

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

ORIGINAL_BLOCK_ALLOCATION = '''    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> None:
'''

PATCHED_BLOCK_ALLOCATION = '''    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _publish(self, block: Block, hash: int, token_ids: list[int | bytes]) -> None:
        block.update(hash, token_ids)
        self.hash_to_block_id[hash] = block.block_id

    def _deallocate_block(self, block_id: int) -> None:
'''

ORIGINAL_ALLOCATE_PUBLISH = '''            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
'''

PATCHED_ALLOCATE_PUBLISH = '''            if h != -1:
                self._publish(block, h, token_ids)
            seq.block_table.append(block_id)
'''

ORIGINAL_APPEND_PUBLISH = '''                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
'''

PATCHED_APPEND_PUBLISH = '''                h = self.compute_hash(token_ids, prefix)
                self._publish(last_block, h, token_ids)
        else:
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
    block_manager = root / "engine" / "block_manager.py"
    tokenizer = root / "models" / "voxcpm2" / "utils.py"

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
    block_manager_source = block_manager.read_text(encoding="utf-8")
    block_manager_source, block_allocation_changed = replace_once(
        block_manager_source,
        ORIGINAL_BLOCK_ALLOCATION,
        PATCHED_BLOCK_ALLOCATION,
        "block hash eviction",
    )
    block_manager_source, allocate_publish_changed = replace_once(
        block_manager_source,
        ORIGINAL_ALLOCATE_PUBLISH,
        PATCHED_ALLOCATE_PUBLISH,
        "allocation hash publication",
    )
    block_manager_source, append_publish_changed = replace_once(
        block_manager_source,
        ORIGINAL_APPEND_PUBLISH,
        PATCHED_APPEND_PUBLISH,
        "append hash publication",
    )
    block_manager_changed = any(
        (block_allocation_changed, allocate_publish_changed, append_publish_changed)
    )
    tokenizer_source, tokenizer_changed = replace_once(
        tokenizer.read_text(encoding="utf-8"),
        ORIGINAL_CHINESE_TOKEN_LOOKUP,
        PATCHED_CHINESE_TOKEN_LOOKUP,
        "Chinese tokenizer lookup",
    )
    if not any((allocator_cleanup_changed, memory_error_changed, block_manager_changed, tokenizer_changed)):
        print(f"{root} is already patched")
        return

    if allocator_cleanup_changed:
        runner.write_text(runner_source, encoding="utf-8")
    if memory_error_changed:
        engine.write_text(engine_source, encoding="utf-8")
    if block_manager_changed:
        block_manager.write_text(block_manager_source, encoding="utf-8")
    if tokenizer_changed:
        tokenizer.write_text(tokenizer_source, encoding="utf-8")
    print(f"Patched Nano-vLLM runtime behavior in {root}")


if __name__ == "__main__":
    main()
