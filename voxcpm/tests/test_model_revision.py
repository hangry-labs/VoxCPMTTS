from unittest.mock import patch

from voxcpm.core import VoxCPM
from voxcpm.nano_backend import NanoVoxCPM


def test_native_model_download_uses_requested_revision() -> None:
    with (
        patch("voxcpm.core.snapshot_download", return_value="/models/voxcpm2") as download,
        patch.object(VoxCPM, "__init__", return_value=None),
    ):
        VoxCPM.from_pretrained(
            hf_model_id="openbmb/VoxCPM2",
            revision="model-commit",
            local_files_only=True,
            load_denoiser=False,
            optimize=False,
        )

    download.assert_called_once_with(
        repo_id="openbmb/VoxCPM2",
        cache_dir=None,
        local_files_only=True,
        revision="model-commit",
    )


def test_nano_model_download_uses_requested_revision() -> None:
    with (
        patch("voxcpm.nano_backend.snapshot_download", return_value="/models/voxcpm2") as download,
        patch.object(NanoVoxCPM, "__init__", return_value=None),
    ):
        NanoVoxCPM.from_pretrained(
            hf_model_id="openbmb/VoxCPM2",
            revision="model-commit",
            local_files_only=True,
        )

    download.assert_called_once_with(
        repo_id="openbmb/VoxCPM2",
        cache_dir=None,
        local_files_only=True,
        revision="model-commit",
    )
