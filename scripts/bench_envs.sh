#!/usr/bin/env bash
# D0 benchmark: train the upstream flat-walking task for a fixed time
# and log the GPU state.
#
# Usage:   scripts/bench_envs.sh NUM_ENVS [DURATION_S] [MAX_ITERATIONS]
# Example: scripts/bench_envs.sh 2048 600
#
# Outputs, in ~/projectD/logs/bench/:
#   train_<NUM_ENVS>.log   training output (iterations, steps/s, reward)
#   gpu_<NUM_ENVS>.csv     GPU temperature, power, utilisation, memory (10 s)
# Checkpoints and TensorBoard files go to the upstream folder, logs/rsl_rl/.

set -uo pipefail

NUM_ENVS="${1:?Give the number of environments}"
DURATION_S="${2:-600}"
MAX_ITERATIONS="${3:-50000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="$HOME/projectD/third_party/microduck_rl"
OUT="$HOME/projectD/logs/bench"
mkdir -p "$OUT"
rm -f "$OUT/gpu_${NUM_ENVS}.csv"

# GPU monitor in the background. Sample every 10 s, warn above 87 C.
bash "$HERE/gpu_monitor.sh" "$OUT/gpu_${NUM_ENVS}.csv" 10 87 &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null' EXIT

cd "$UPSTREAM"
# `timeout -s INT` sends Ctrl+C after DURATION_S seconds.
# The TensorBoard logger keeps all data on this computer.
timeout -s INT -k 30 "$DURATION_S" \
    uv run --locked train Mjlab-Velocity-Flat-MicroDuck \
        --env.scene.num-envs "$NUM_ENVS" \
        --agent.logger tensorboard \
        --agent.run-name "bench_${NUM_ENVS}" \
        --agent.max-iterations "$MAX_ITERATIONS" \
        --agent.seed 0 \
    > "$OUT/train_${NUM_ENVS}.log" 2>&1
echo "training exit code: $? (124 means the time limit stopped it)"
