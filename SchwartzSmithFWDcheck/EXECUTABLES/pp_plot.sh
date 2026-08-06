#!/bin/sh
# EXECUTABLES/pp_plot.sh
# Args passed from pp_plot.sub's `arguments` line, in this exact order:
#   $1 results_dir   $2 out_path

RESULTS_DIR=$1
OUT_PATH=$2

echo "=========================================="
echo " Stage 3 Layer 1 PP plot"
echo " results_dir = ${RESULTS_DIR}"
echo "=========================================="

python /home/2673888s/commodity_futures/SchwartzSmithFWDcheck/collect_pp_plot.py \
    --results_dir "${RESULTS_DIR}" \
    --out_path "${OUT_PATH}"