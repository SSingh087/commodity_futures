#!/bin/bash

N_CHUNKS=$1
CONFIG=$2

echo "================================================"
echo " merge_options_data  (${N_CHUNKS} chunks)"
echo " config: ${CONFIG}"
echo "================================================"

python generate_training_data/merge_options_data.py \
    --config   "${CONFIG}"   \
    --n_chunks "${N_CHUNKS}"