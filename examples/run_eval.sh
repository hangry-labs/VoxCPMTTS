#!/usr/bin/env bash

# Placeholder for a future VoxCPM2 evaluation pipeline.

set -euo pipefail

cat <<'MSG'
No benchmark evaluation harness is wired for VoxCPMTTS yet.

For a smoke inference run after installing dependencies, use:

  python -m voxcpm.cli design --text "Hello from VoxCPM2." --output out.wav --no-denoiser --no-optimize

MSG
