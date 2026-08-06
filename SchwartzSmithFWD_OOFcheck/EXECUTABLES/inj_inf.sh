#!/bin/sh
# EXECUTABLES/inj_inf.sh  (SchwartzSmithFWD_OOFcheck)
# Args in order: $1 checkpoint_dir  $2 seed  $3 output_root
#                $4 n_contracts     $5 n_live  $6 n_pool

CHECKPOINT_DIR=$1
SEED=$2
OUTPUT_ROOT=$3
N_CONTRACTS=$4
N_LIVE=$5
N_POOL=$6

echo "=========================================="
echo " Stage 3 Layer 2 (options) injection"
echo " model = $(basename ${CHECKPOINT_DIR})"
echo " seed  = ${SEED}"
echo "=========================================="

python /home/2673888s/commodity_futures/SchwartzSmithFWD_OOFcheck/run_inference.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --seed "${SEED}" \
    --output_root "${OUTPUT_ROOT}" \
    --n_contracts "${N_CONTRACTS}" \
    --n_live "${N_LIVE}" \
    --n_pool "${N_POOL}"