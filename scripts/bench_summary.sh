#!/usr/bin/env bash
# Summarise one D0 benchmark run made by bench_envs.sh.
#
# Usage: scripts/bench_summary.sh NUM_ENVS

set -uo pipefail
N="${1:?Give the number of environments}"
OUT="$HOME/projectD/logs/bench"
LOG="$OUT/train_${N}.log"
GPU="$OUT/gpu_${N}.csv"

# Print the value on the last line that contains the label $1.
last() { grep -a "$1" "$LOG" | tail -1 | sed 's/^[^:]*: *//'; }
# Print the value at iteration $2 (the $2-th time the label $1 occurs).
at() { grep -a "$1" "$LOG" | sed -n "${2}p" | sed 's/.*: *//'; }

iters=$(grep -a -c "Learning iteration" "$LOG")
echo "== ${N} environments =="
echo "iterations done:      $iters"
echo "time elapsed:         $(last 'Time elapsed')"
echo "iteration time:       $(last 'Iteration time')"
echo "steps per second:     median $(grep -a 'Steps per second' "$LOG" | sed 's/.*: *//' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')"
for i in 1 50 100 200 "$iters"; do
    [ "$i" -le "$iters" ] || continue
    echo "iteration $i: reward $(at 'Mean reward' "$i"), episode length $(at 'Mean episode length' "$i"), fell_over $(at 'Termination/fell_over' "$i"), error_vel_xy $(at 'error_vel_xy' "$i")"
done
echo "errors in log:        $(grep -a -c -i -E 'out of memory|nan_state: [1-9]' "$LOG")"

# GPU columns: time,temp_c,power_w,util_pct,mem_used_mib,mem_total_mib
# Skip the header and the first 6 samples (60 s of start-up).
tail -n +8 "$GPU" | awk -F, '
    { t+=$2; p+=$3; u+=$4; n++
      if ($2>tm) tm=$2; if ($3>pm) pm=$3; if ($5>mm) mm=$5 }
    END { printf "GPU temp: mean %.0f C, max %d C | power: mean %.0f W, max %.0f W | util: mean %.0f %% | memory max: %d MiB (%d samples)\n",
          t/n, tm, p/n, pm, u/n, mm, n }'
