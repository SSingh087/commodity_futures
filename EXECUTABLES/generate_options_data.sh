#!/bin/bash

CHUNK_ID=$1
N_CHUNKS=$2
CONFIG=$3

echo "================================================"
echo " gen_options_data  chunk ${CHUNK_ID} / ${N_CHUNKS}"
echo " config: ${CONFIG}"
echo "================================================"

python generate_training_data/SchwartzSmithModel_OOF.py \
    --config  "${CONFIG}"       \
    --job_id  "${CHUNK_ID}"     \
    --n_jobs  "${N_CHUNKS}"