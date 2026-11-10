#!/usr/bin/env bash
# Wait until a training run ends. Exit early when the GPU is too hot.
#
# Usage: scripts/watch_train.sh RUN_NAME GPU_CSV [TEMP_LIMIT_C]
# Exit codes: 0 training ended, 2 GPU above the limit for 3 samples in a row.

RUN_NAME="${1:?Give the run name}"
GPU_CSV="${2:?Give the GPU CSV file}"
LIMIT="${3:-87}"

sleep 120   # Let the training start.
while pgrep -f "run-name ${RUN_NAME}" > /dev/null; do
    # Column 2 is the temperature. Look at the last 3 samples (3 minutes).
    hot=$(tail -n 3 "$GPU_CSV" | awk -F, -v l="$LIMIT" '$2>l {n++} END {print n+0}')
    if [ "$hot" -ge 3 ]; then
        echo "GPU above ${LIMIT} C for 3 samples:"; tail -n 3 "$GPU_CSV"
        exit 2
    fi
    sleep 60
done
echo "training ended"
