#!/bin/bash
set -euo pipefail

# CondFiLM-R-NoDist base->new sweep over the standard 11 datasets.
# Usage:
#   bash scripts/token_mod/run_condfilm_11datasets.sh
# Optional overrides:
#   DATA=/path/to/dataset SEED=1 MAX_EPOCH=40 bash scripts/token_mod/run_condfilm_11datasets.sh

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${DATA:-/home/ouyangjun/workspace/data/a/shiXiao/dataset}"
SEED="${SEED:-1}"
MAX_EPOCH="${MAX_EPOCH:-40}"
SHOTS="${SHOTS:-16}"
CFG="${CFG:-vit_b16_ep50_k16}"
TRAINER=TokenModHiCroPL
TEST_BS="${TEST_BS:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRINT_FREQ="${PRINT_FREQ:-20}"

DATASETS=(
  stanford_cars
  caltech101
  dtd
  eurosat
  fgvc_aircraft
  food101
  oxford_flowers
  oxford_pets
  sun397
  ucf101
  imagenet
)

RUN_TAG="seed${SEED}_condfilm_r_nodist_e${MAX_EPOCH}"
LOG_DIR="output/token_mod/logs"
mkdir -p "${LOG_DIR}"

echo "[TokenMod] DATA=${DATA}"
echo "[TokenMod] SEED=${SEED} MAX_EPOCH=${MAX_EPOCH} SHOTS=${SHOTS}"
echo "[TokenMod] datasets=${DATASETS[*]}"

for DATASET in "${DATASETS[@]}"; do
  TRAIN_DIR="output/token_mod/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${RUN_TAG}"
  TEST_DIR="output/token_mod/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${RUN_TAG}"
  TRAIN_LOG="${LOG_DIR}/${DATASET}_${RUN_TAG}_train.out.txt"
  TEST_LOG="${LOG_DIR}/${DATASET}_${RUN_TAG}_novel.out.txt"

  echo "============================================================"
  echo "[TokenMod] TRAIN ${DATASET}"
  echo "  out: ${TRAIN_DIR}"
  echo "  log: ${TRAIN_LOG}"
  echo "============================================================"

  python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${TRAIN_DIR}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES base \
    OPTIM.MAX_EPOCH "${MAX_EPOCH}" \
    TRAIN.PRINT_FREQ "${PRINT_FREQ}" \
    DATALOADER.TEST.BATCH_SIZE "${TEST_BS}" \
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
    TRAINER.TOKENMOD.MODULATION film \
    TRAINER.TOKENMOD.USE_RESIDUAL True \
    TRAINER.TOKENMOD.USE_DISTILL False \
    TRAINER.TOKENMOD.RESIDUAL_GATE False \
    TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True \
    2>&1 | tee "${TRAIN_LOG}"

  echo "============================================================"
  echo "[TokenMod] NOVEL ${DATASET}"
  echo "  model: ${TRAIN_DIR}"
  echo "  out: ${TEST_DIR}"
  echo "  log: ${TEST_LOG}"
  echo "============================================================"

  python train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${TEST_DIR}" \
    --model-dir "${TRAIN_DIR}" \
    --eval-only \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES new \
    DATALOADER.TEST.BATCH_SIZE "${TEST_BS}" \
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
    TRAINER.TOKENMOD.MODULATION film \
    TRAINER.TOKENMOD.USE_RESIDUAL True \
    TRAINER.TOKENMOD.USE_DISTILL False \
    TRAINER.TOKENMOD.RESIDUAL_GATE False \
    TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True \
    2>&1 | tee "${TEST_LOG}"
done

echo "[TokenMod] All 11 datasets finished."
