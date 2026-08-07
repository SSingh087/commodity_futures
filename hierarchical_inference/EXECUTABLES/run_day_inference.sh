#!/bin/sh
# EXECUTABLES/day_inf.sh
# Args passed from day_inf.sub's `arguments` line, in this exact order:
#   $1 checkpoint_dir   $2 day_dir   $3 output_root   $4 n_live   $5 n_pool

CHECKPOINT_DIR=$1
DAY_DIR=$2
OUTPUT_ROOT=$3
N_LIVE=$4
N_POOL=$5

echo "=========================================="
echo " per-day inference — $(basename ${DAY_DIR})"
echo "=========================================="

python /home/2673888s/commodity_futures/hierarchical_inference/run_day_inference.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --day_dir "${DAY_DIR}" \
    --output_root "${OUTPUT_ROOT}" \
    --n_live "${N_LIVE}" \
    --n_pool "${N_POOL}"