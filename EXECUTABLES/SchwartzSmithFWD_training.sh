#!/bin/sh
# EXECUTABLES/SchwartzSmithFWD_training.sh
# Args passed from the submit file's `arguments` line, in this exact order:
#   $1 batch_size   $2 alpha        $3 beta          $4 n_epochs
#   $5 patience     $6 lr           $7 weight_decay  $8 tag (Cluster_Process)

BATCH_SIZE=$1
ALPHA=$2
BETA=$3
N_EPOCHS=$4
PATIENCE=$5
LR=$6
WEIGHT_DECAY=$7
TAG=$8

echo "=========================================="
echo " SchwartzSmithFWD training — job tag ${TAG}"
echo " batch_size=${BATCH_SIZE}  alpha=${ALPHA}  beta=${BETA}"
echo " n_epochs=${N_EPOCHS}  patience=${PATIENCE}"
echo " lr=${LR}  weight_decay=${WEIGHT_DECAY}"
echo "=========================================="

python /home/2673888s/commodity_futures/generate_training_data/SchwartzSmithModelFWD_train.py \
    --config       /home/2673888s/commodity_futures/config/SchwartzSmithFWD.yaml \
    --batch_size   "${BATCH_SIZE}" \
    --alpha        "${ALPHA}" \
    --beta         "${BETA}" \
    --n_epochs     "${N_EPOCHS}" \
    --patience     "${PATIENCE}" \
    --lr           "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --tag          "${TAG}"