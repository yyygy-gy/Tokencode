#!/bin/bash
DATA="E:/Project/dataset"
TRAINER=TokenModHiCroPL
DATASET=$1
SEED=$2
CFG=vit_b16_ep50_k16
SHOTS=16
SUB=new
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/token_mod/train_base/${COMMON_DIR}
DIR=output/token_mod/test_${SUB}/${COMMON_DIR}
python train.py \
  --root ${DATA} \
  --seed ${SEED} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
  --output-dir ${DIR} \
  --model-dir ${MODEL_DIR} \
  --eval-only \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES ${SUB}
