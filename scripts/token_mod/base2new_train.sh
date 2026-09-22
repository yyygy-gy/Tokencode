#!/bin/bash
DATA="E:/Project/dataset"
TRAINER=TokenModHiCroPL
DATASET=$1
SEED=$2
CFG=vit_b16_ep50_k16
SHOTS=16
DIR=output/token_mod/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
python train.py \
  --root ${DATA} \
  --seed ${SEED} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
  --output-dir ${DIR} \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES base
