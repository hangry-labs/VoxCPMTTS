import json

from voxcpm import core


def test_saved_lora_config_is_loaded_beside_checkpoint(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architecture":"voxcpm2"}', encoding="utf-8")
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    (lora_dir / "lora_weights.ckpt").write_bytes(b"checkpoint")
    (lora_dir / "lora_config.json").write_text(
        json.dumps(
            {
                "lora_config": {
                    "enable_lm": True,
                    "enable_dit": True,
                    "enable_proj": False,
                    "r": 32,
                    "alpha": 64,
                    "dropout": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    class FakeModel:
        lora_config = None
        sample_rate = 48_000

        def load_lora_weights(self, path):
            observed["weights"] = path
            return [], []

    def fake_from_local(path, *, optimize, device, lora_config):
        observed["config"] = lora_config
        return FakeModel()

    monkeypatch.setattr(core.VoxCPM2Model, "from_local", fake_from_local)

    core.VoxCPM(
        str(model_dir),
        enable_denoiser=False,
        optimize=False,
        lora_weights_path=str(lora_dir),
    )

    assert observed["config"].r == 32
    assert observed["config"].alpha == 64
    assert observed["config"].dropout == 0.1
    assert observed["weights"] == str(lora_dir)
