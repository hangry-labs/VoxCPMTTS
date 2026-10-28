# Modified by Hangry Labs in 2026 for the Docker-first inference fork.

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "voxcpm" / "cli.py"
V1_MODEL_PATH = ROOT / "models" / "openbmb__VoxCPM1.5"
V2_MODEL_PATH = ROOT / "models" / "VoxCPM2-1B-newaudiovae-6hz-nope-sft"


pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

spec = importlib.util.spec_from_file_location("voxcpm.cli", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
sys.modules["voxcpm.cli"] = cli
assert spec.loader is not None
spec.loader.exec_module(cli)


class DummyTTSModel:
    sample_rate = 16000


class DummyModel:
    def __init__(self):
        self.tts_model = DummyTTSModel()
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(160, dtype=np.float32)


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["voxcpm", *argv])
    cli.main()


def patch_soundfile_write(monkeypatch):
    monkeypatch.setattr(cli, "sf", types.SimpleNamespace(write=lambda *args, **kwargs: None))


def test_parser_defaults_to_voxcpm2():
    parser = cli._build_parser()
    args = parser.parse_args(["design", "--text", "hello", "--output", "out.wav"])
    assert args.hf_model_id == "openbmb/VoxCPM2"
    assert args.device == "auto"
    assert args.no_optimize is False


def test_cli_heavy_dependencies_are_lazy():
    assert cli.VoxCPM is None
    assert cli.sf is None


def test_load_model_respects_no_optimize_for_local_model(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            self.tts_model = DummyTTSModel()

    monkeypatch.setattr(cli, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
            "--model-path",
            str(V2_MODEL_PATH),
            "--no-optimize",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "auto"
    assert calls["kwargs"]["optimize"] is False


def test_load_model_defaults_optimize_for_hf(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["kwargs"] = kwargs
            return DummyModel()

    monkeypatch.setattr(cli, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "auto"
    assert calls["kwargs"]["optimize"] is True


def test_load_model_respects_no_optimize_for_hf(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["kwargs"] = kwargs
            return DummyModel()

    monkeypatch.setattr(cli, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
            "--no-optimize",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "auto"
    assert calls["kwargs"]["optimize"] is False


def test_load_model_passes_explicit_device_to_hf(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["kwargs"] = kwargs
            return DummyModel()

    monkeypatch.setattr(cli, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
            "--device",
            "mps",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "mps"


def test_design_subcommand_applies_control(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "design",
            "--text",
            "hello",
            "--control",
            "warm female voice",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert dummy_model.calls[0]["text"] == "(warm female voice)hello"
    assert dummy_model.calls[0]["prompt_wav_path"] is None
    assert dummy_model.calls[0]["reference_wav_path"] is None


def test_clone_subcommand_reads_prompt_file(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt transcript\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert dummy_model.calls[0]["prompt_wav_path"] == str(prompt_audio)
    assert dummy_model.calls[0]["prompt_text"] == "prompt transcript"


def test_clone_rejects_reference_audio_for_v1_local_model(monkeypatch, tmp_path):
    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            str(reference_audio),
            "--model-path",
            str(V1_MODEL_PATH),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()


def test_clone_rejects_reference_audio_for_v1_hf_model_id(monkeypatch, tmp_path):
    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            str(reference_audio),
            "--hf-model-id",
            "openbmb/VoxCPM1.5",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()


def test_legacy_root_args_still_work_and_warn(monkeypatch, tmp_path, capsys):
    dummy_model = DummyModel()
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "--text",
            "hello",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert dummy_model.calls[0]["text"] == "hello"


def test_batch_subcommand_applies_control(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\nworld\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path / "outs"),
            "--control",
            "calm narrator",
        ],
    )

    assert [call["text"] for call in dummy_model.calls] == [
        "(calm narrator)hello",
        "(calm narrator)world",
    ]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "scenario",
    ["success", "generation", "all-generation", "audio-write", "timestamp-write", "alignment-warning", "alignment-strict"],
)
def test_batch_completion_status_and_retained_outputs(monkeypatch, tmp_path, capsys, legacy, scenario):
    import soundfile
    import voxcpm.timestamps as timestamps

    texts = ["first", "second", "third"]
    input_file = tmp_path / "texts.txt"
    input_file.write_text("\n".join(texts) + "\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    attempted = []

    class BatchModel(DummyModel):
        def generate(self, **kwargs):
            attempted.append(kwargs["text"])
            if scenario == "all-generation" or (scenario == "generation" and kwargs["text"] == "second"):
                raise RuntimeError("Controlled generation failure")
            return super().generate(**kwargs)

    def write_audio(path, audio, sample_rate):
        if scenario == "audio-write" and Path(path).stem == "output_002":
            raise OSError("Controlled audio write failure")
        soundfile.write(path, audio, sample_rate)

    def align_audio(**kwargs):
        if scenario in {"alignment-warning", "alignment-strict"} and kwargs["text"] == "second":
            raise RuntimeError("Controlled alignment failure")
        return {"text": kwargs["text"], "items": []}

    monkeypatch.setattr(cli, "load_model", lambda args: BatchModel())
    monkeypatch.setattr(cli, "sf", types.SimpleNamespace(write=write_audio))
    monkeypatch.setattr(timestamps, "align_audio_file", align_audio)
    argv = [] if legacy else ["batch"]
    argv += ["--input", str(input_file), "--output-dir", str(output_dir)]
    if scenario.startswith("alignment") or scenario == "timestamp-write":
        argv.append("--timestamps")
    if scenario == "alignment-strict":
        argv.append("--timestamp-strict")
    if scenario == "timestamp-write":
        output_dir.mkdir()
        (output_dir / "output_002.timestamps.json").mkdir()

    failed_job = scenario in {"generation", "all-generation", "audio-write", "timestamp-write"}
    if failed_job:
        with pytest.raises(SystemExit) as failure:
            run_main(monkeypatch, argv)
        assert failure.value.code == 1
    elif scenario == "alignment-strict":
        with pytest.raises(SystemExit, match="Timestamp alignment failed"):
            run_main(monkeypatch, argv)
    else:
        run_main(monkeypatch, argv)

    assert attempted == (texts[:2] if scenario == "alignment-strict" else texts)
    expected_indices = [1, 2, 3]
    if scenario == "all-generation":
        expected_indices = []
    elif scenario in {"generation", "audio-write"}:
        expected_indices = [1, 3]
    elif scenario == "alignment-strict":
        expected_indices = [1, 2]
    outputs = sorted(output_dir.glob("*.wav"))
    assert [path.name for path in outputs] == [f"output_{index:03d}.wav" for index in expected_indices]
    for output in outputs:
        info = soundfile.info(output)
        assert (info.frames, info.samplerate, info.channels) == (160, 16000, 1)
    stderr = capsys.readouterr().err
    if scenario == "alignment-strict":
        assert "Batch finished:" not in stderr
    else:
        completed = 0 if scenario == "all-generation" else 2 if failed_job else 3
        assert f"Batch finished: {completed}/3 succeeded" in stderr
    if scenario == "alignment-warning":
        assert "Warning: Timestamp alignment failed" in stderr


def test_legacy_clone_with_prompt_file_still_works(monkeypatch, tmp_path, capsys):
    dummy_model = DummyModel()
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("legacy transcript", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert dummy_model.calls[0]["prompt_text"] == "legacy transcript"


def test_invalid_prompt_text_and_prompt_file_combination(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("transcript", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "inline transcript",
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "Use either --prompt-text or --prompt-file" in capsys.readouterr().err


def test_missing_prompt_file_reports_parser_error(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-file",
            str(tmp_path / "missing.txt"),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "prompt text file" in capsys.readouterr().err


def test_design_rejects_prompt_audio_args(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "design",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "transcript",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "does not accept prompt/reference audio" in capsys.readouterr().err


def test_clone_rejects_prompt_audio_without_transcript(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert (
        "--prompt-audio requires --prompt-text or --prompt-file"
        in capsys.readouterr().err
    )


def test_clone_rejects_transcript_without_prompt_audio(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-text",
            "transcript",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert (
        "--prompt-text/--prompt-file requires --prompt-audio" in capsys.readouterr().err
    )


def test_batch_rejects_control_with_prompt_transcript(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\n", encoding="utf-8")
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path / "outs"),
            "--control",
            "calm narrator",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "transcript",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "--control cannot be used together" in capsys.readouterr().err


def test_detect_model_architecture_uses_local_configs():
    parser = cli._build_parser()
    v1_args = parser.parse_args(
        [
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--model-path",
            str(V1_MODEL_PATH),
            "--output",
            "out.wav",
        ]
    )
    v2_args = parser.parse_args(
        [
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--model-path",
            str(V2_MODEL_PATH),
            "--output",
            "out.wav",
        ]
    )

    assert cli.detect_model_architecture(v1_args) == "voxcpm"
    assert cli.detect_model_architecture(v2_args) == "voxcpm2"


def test_parser_accepts_generation_seed():
    args = cli._build_parser().parse_args(
        ["design", "--text", "hello", "--output", "out.wav", "--seed", "42"]
    )

    assert args.seed == 42


def test_design_subcommand_passes_seed(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "design",
            "--text",
            "hello",
            "--seed",
            "123",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert dummy_model.calls[0]["seed"] == 123


def test_design_writes_timestamp_sidecar_when_requested(monkeypatch, tmp_path):
    import voxcpm.timestamps as timestamps

    dummy_model = DummyModel()
    output = tmp_path / "out.wav"
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)
    monkeypatch.setattr(
        timestamps,
        "align_audio_file",
        lambda **kwargs: {
            "audio_path": kwargs["audio_path"],
            "sample_rate": kwargs["sample_rate"],
            "backend": kwargs["backend"],
            "level": kwargs["level"],
            "text": kwargs["text"],
            "items": [{"text": "hello", "start": 0.0, "end": 0.5, "level": "word"}],
            "warning": None,
        },
    )

    run_main(
        monkeypatch,
        ["design", "--text", "hello", "--output", str(output), "--timestamps"],
    )

    sidecar = json.loads((tmp_path / "out.timestamps.json").read_text(encoding="utf-8"))
    assert sidecar["level"] == "word"
    assert sidecar["items"][0]["text"] == "hello"
