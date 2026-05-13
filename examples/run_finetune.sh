#!/usr/bin/env bash

# Convenience wrapper for VoxCPM2 fine-tuning on this fork.

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-conf/voxcpm_v2/voxcpm_finetune_lora.yaml}"

cd "$(dirname "$0")/.."

echo "Using config: ${CONFIG_PATH}"
echo "Edit the YAML before running on real data."

python scripts/train_voxcpm_finetune.py --config_path "${CONFIG_PATH}"
