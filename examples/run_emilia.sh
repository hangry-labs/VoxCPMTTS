#!/usr/bin/env bash

# Placeholder for a future VoxCPM2 Emilia training pipeline.
# The previous WebDataset pipeline copied from another project does not apply to this fork.

set -euo pipefail

cat <<'MSG'
No Emilia end-to-end pipeline is wired for VoxCPMTTS yet.

For the current training path, prepare a VoxCPM JSONL manifest and run:

  python -m voxcpm.cli validate --manifest path/to/train.jsonl
  python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

MSG
