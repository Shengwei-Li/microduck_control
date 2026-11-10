#!/usr/bin/env bash
# Show a trained walking policy in the web viewer (viser).
# Open http://localhost:8080 in a browser on Windows.
#
# Usage:   scripts/play_local.sh CHECKPOINT [NUM_ENVS]
# Example: scripts/play_local.sh ~/projectD/third_party/microduck_rl/logs/rsl_rl/velocity/2026-11-10_13-27-00_p0_seed0/model_2999.pt 4
#
# The upstream `play` reads from wandb by default. --checkpoint-file makes it
# read a local file instead. Stop it with Ctrl+C.

set -euo pipefail
CHECKPOINT="${1:?Give a checkpoint file (model_N.pt)}"
NUM_ENVS="${2:-4}"

cd "$HOME/projectD/third_party/microduck_rl"
uv run --locked play Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file "$CHECKPOINT" \
    --num-envs "$NUM_ENVS" \
    --viewer viser
