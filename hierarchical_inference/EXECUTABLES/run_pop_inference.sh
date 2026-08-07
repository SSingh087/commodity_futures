#!/bin/sh
# EXECUTABLES/hier_inf.sh
# Args passed from hier_inf.sub's `arguments` line, in this exact order:
#   $1 day_posteriors_root   $2 lambda_true_path   $3 output_root   $4 n_live   $5 n_pool

DAY_POSTERIORS_ROOT=$1
LAMBDA_TRUE_PATH=$2
OUTPUT_ROOT=$3
N_LIVE=$4
N_POOL=$5

echo "=========================================="
echo " hierarchical inference"
echo "=========================================="

python /home/2673888s/commodity_futures/hierarchical_inference/run_pop_inference.py \
    --day_posteriors_root "${DAY_POSTERIORS_ROOT}" \
    --lambda_true_path "${LAMBDA_TRUE_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --n_live "${N_LIVE}" \
    --n_pool "${N_POOL}"