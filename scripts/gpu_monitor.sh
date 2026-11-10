#!/usr/bin/env bash
# Log GPU temperature, power, utilisation and memory to a CSV file.
#
# Usage:   scripts/gpu_monitor.sh LOG_FILE [INTERVAL_S] [TEMP_LIMIT_C]
# Example: scripts/gpu_monitor.sh logs/gpu_bench.csv 60 87
#
# The script prints a warning line when the temperature is above the limit
# for 3 samples in a row. It does not stop the training. Stop it with Ctrl+C.

set -euo pipefail

LOG_FILE="${1:?Give a log file path}"
INTERVAL_S="${2:-60}"
TEMP_LIMIT_C="${3:-87}"

mkdir -p "$(dirname "$LOG_FILE")"
if [ ! -f "$LOG_FILE" ]; then
    echo "time,temp_c,power_w,util_pct,mem_used_mib,mem_total_mib" > "$LOG_FILE"
fi

hot_count=0
while true; do
    # One line, for example: "71, 68.21, 97, 5321, 8151"
    line=$(nvidia-smi \
        --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits)
    echo "$(date +%Y-%m-%dT%H:%M:%S),${line// /}" >> "$LOG_FILE"

    temp=${line%%,*}
    if [ "$temp" -gt "$TEMP_LIMIT_C" ]; then
        hot_count=$((hot_count + 1))
    else
        hot_count=0
    fi
    if [ "$hot_count" -ge 3 ]; then
        echo "WARNING $(date +%H:%M:%S): GPU at ${temp} C, above ${TEMP_LIMIT_C} C for ${hot_count} samples" >&2
    fi

    sleep "$INTERVAL_S"
done
