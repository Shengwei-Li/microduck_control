#!/usr/bin/env bash
# D3: continue the training of P0 on a payload task (fine-tuning).
#
# Usage:   scripts/finetune.sh TASK SEED [ITERATIONS] [NUM_ENVS]
# Example: scripts/finetune.sh PayloadDuck-P1 0 1000 4096
#
# The run starts from the last checkpoint of P0. The upstream resume keeps the
# iteration counter, so the curricula stay at their final stage and the new
# checkpoints are model_3000.pt ... model_<2999 + ITERATIONS>.pt.
#
# Outputs:
#   ~/projectD/logs/d3/<TASK>_seed<SEED>.log        training output
#   ~/projectD/logs/d3/<TASK>_seed<SEED>_gpu.csv    GPU state every 60 s
#   upstream logs/rsl_rl/velocity/<date>_<TASK>_seed<SEED>/   checkpoints, TensorBoard

set -uo pipefail

TASK="$1"
SEED="$2"
ITERATIONS="${3:-1000}"
NUM_ENVS="${4:-4096}"
P0_RUN="${P0_RUN:-2026-11-10_13-27-00_p0_seed0}"
P0_CHECKPOINT="${P0_CHECKPOINT:-model_2999.pt}"

HERE="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="$HOME/projectD/third_party/microduck_rl"
OUT="$HOME/projectD/logs/d3"
NAME="${TASK}_seed${SEED}"
mkdir -p "$OUT"

bash "$HERE/gpu_monitor.sh" "$OUT/${NAME}_gpu.csv" 60 87 2>> "$OUT/${NAME}_gpu_warnings.log" &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null' EXIT

cd "$UPSTREAM"
uv run --locked python "$HERE/train.py" "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.logger tensorboard \
    --agent.run-name "$NAME" \
    --agent.max-iterations "$ITERATIONS" \
    --agent.seed "$SEED" \
    --env.seed "$SEED" \
    --agent.resume True \
    --agent.load-run "$P0_RUN" \
    --agent.load-checkpoint "$P0_CHECKPOINT" \
    > "$OUT/${NAME}.log" 2>&1
echo "training exit code: $?" >> "$OUT/${NAME}.log"
