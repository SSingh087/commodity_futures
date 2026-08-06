#!/bin/sh
# EXECUTABLES/pp_plot.sh  (SchwartzSmithFWD_OOFcheck)
# $1 results_dir  $2 out_path

RESULTS_DIR=$1
OUT_PATH=$2

echo "=========================================="
echo " Stage 3 Layer 2 (options) PP plot"
echo " results_dir = ${RESULTS_DIR}"
echo "=========================================="

python /home/2673888s/commodity_futures/SchwartzSmithFWD_OOFcheck/collect_pp_plot.py \
    --results_dir "${RESULTS_DIR}" \
    --out_path "${OUT_PATH}"