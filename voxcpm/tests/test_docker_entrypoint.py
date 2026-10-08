from __future__ import annotations

from pathlib import Path

from voxcpm.docker_entrypoint import seed_cache


def test_seed_cache_merges_without_removing_downloaded_files(tmp_path: Path) -> None:
    source = tmp_path / "baked"
    destination = tmp_path / "persistent"
    (source / "snapshots").mkdir(parents=True)
    (destination / "downloads").mkdir(parents=True)
    (source / "snapshots" / "model.bin").write_bytes(b"model")
    (destination / "downloads" / "user.bin").write_bytes(b"user")

    copied_files, copied_bytes = seed_cache(source, destination)
    repeated_files, repeated_bytes = seed_cache(source, destination)

    assert (destination / "snapshots" / "model.bin").read_bytes() == b"model"
    assert (destination / "downloads" / "user.bin").read_bytes() == b"user"
    assert (copied_files, copied_bytes) == (1, 5)
    assert (repeated_files, repeated_bytes) == (0, 0)
