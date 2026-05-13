# VoxCPMTTS Examples

This directory contains the static project page plus upstream-style VoxCPM fine-tuning examples.

## Static Project Page

- `index.html` is the GitHub Pages landing page for Hangry Labs VoxCPMTTS.
- `voices.js` contains the API example cards shown on the page.
- `player.js` and `background.js` support the page UI.

Preview locally by opening `examples/index.html` in a browser.

## Fine-Tuning

VoxCPM supports full fine-tuning and LoRA fine-tuning through the checked-in training script and YAML configs.

Example LoRA run:

```bash
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
```

Example full fine-tune run:

```bash
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml
```

The local manifest validator expects JSONL rows with:

```jsonl
{"text": "Hello world", "audio": "path/to/audio.wav"}
```

Optional `ref_audio` is supported for reference-audio-aware datasets.

Validate before training:

```bash
python -m voxcpm.cli validate --manifest examples/train_data_example.jsonl
```

The copied shell scripts under `examples/` are retained as starting points, but they still need a careful pass before being treated as canonical VoxCPM2 workflows on this fork.
