#!/bin/sh
# EXECUTABLES/inj_inf.sh
# Args passed from inj_inf.sub's `arguments` line, in this exact order:
#   $1 checkpoint_dir   $2 seed   $3 output_root   $4 n_live   $5 n_pool

CHECKPOINT_DIR=$1
SEED=$2
OUTPUT_ROOT=$3
N_LIVE=$4
N_POOL=$5

echo "=========================================="
echo " Stage 3 Layer 1 injection"
echo " model = $(basename ${CHECKPOINT_DIR})"
echo " seed  = ${SEED}"
echo "=========================================="

python /home/2673888s/commodity_futures/SchwartzSmithFWDcheck/run_inference.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --seed "${SEED}" \
    --output_root "${OUTPUT_ROOT}" \
    --n_live "${N_LIVE}" \
    --n_pool "${N_POOL}"