#!/usr/bin/env bash
# D1: full training of the nominal policy P0 (no payload).
#
# Usage:   scripts/train_p0.sh [NUM_ENVS] [MAX_ITERATIONS] [SEED]
# Example: scripts/train_p0.sh 4096 3000 0
#
# Outputs:
#   ~/projectD/logs/p0/train_seed<SEED>.log   training output
#   ~/projectD/logs/p0/gpu_seed<SEED>.csv     GPU state every 60 s
#   upstream logs/rsl_rl/velocity/<date>_p0_seed<SEED>/   checkpoints, TensorBoard
#
# To continue a stopped run, use the upstream flags
#   --agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt

set -uo pipefail

NUM_ENVS="${1:-4096}"
MAX_ITERATIONS="${2:-3000}"
SEED="${3:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="$HOME/projectD/third_party/microduck_rl"
OUT="$HOME/projectD/logs/p0"
mkdir -p "$OUT"

bash "$HERE/gpu_monitor.sh" "$OUT/gpu_seed${SEED}.csv" 60 87 2>> "$OUT/gpu_warnings_seed${SEED}.log" &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null' EXIT

cd "$UPSTREAM"
uv run --locked train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.logger tensorboard \
    --agent.run-name "p0_seed${SEED}" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    --agent.seed "$SEED" \
    --env.seed "$SEED" \
    > "$OUT/train_seed${SEED}.log" 2>&1
echo "training exit code: $?" >> "$OUT/train_seed${SEED}.log"
