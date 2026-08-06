#!/bin/sh
# EXECUTABLES/options_train.sh
# Args passed from the submit file's `arguments` line, in this exact order:
#   $1 n_ensemble  $2 batch_size  $3 n_epochs  $4 patience
#   $5 lr          $6 weight_decay  $7 tag (Cluster_Process)

N_ENSEMBLE=$1
BATCH_SIZE=$2
N_EPOCHS=$3
PATIENCE=$4
LR=$5
WEIGHT_DECAY=$6
TAG=$7

echo "=========================================="
echo " Schwartz-Smith model training — job tag ${TAG}"
echo " n_ensemble=${N_ENSEMBLE}  batch_size=${BATCH_SIZE}"
echo " n_epochs=${N_EPOCHS}  patience=${PATIENCE}"
echo " lr=${LR}  weight_decay=${WEIGHT_DECAY}"
echo "=========================================="

python /home/2673888s/commodity_futures/generate_training_data/SchwartzSmithModel_OOF_train.py \
    --config       /home/2673888s/commodity_futures/config/SchwartzSmithFWD.yaml \
    --n_ensemble   "${N_ENSEMBLE}" \
    --batch_size   "${BATCH_SIZE}" \
    --n_epochs     "${N_EPOCHS}" \
    --patience     "${PATIENCE}" \
    --lr           "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --tag          "${TAG}"