#!/bin/bash
set -euo pipefail

# CondFiLM + task_text + preserve_norm + distill
# base->new sweep over the standard 11 datasets.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/token_mod/run_condfilm_tasktext_preservenorm_distill_11datasets.sh
#
# Optional overrides:
#   DATA=/path/to/dataset SEED=1 MAX_EPOCH=40 CUDA_VISIBLE_DEVICES=3 \
#     bash scripts/token_mod/run_condfilm_tasktext_preservenorm_distill_11datasets.sh
#
# Resume / skip finished runs:
#   SKIP_DONE=1 CUDA_VISIBLE_DEVICES=3 \
#     bash scripts/token_mod/run_condfilm_tasktext_preservenorm_distill_11datasets.sh

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

DATA="${DATA:-/home/ouyangjun/workspace/data/a/shiXiao/dataset}"
SEED="${SEED:-1}"
MAX_EPOCH="${MAX_EPOCH:-40}"
SHOTS="${SHOTS:-16}"
CFG="${CFG:-vit_b16_ep50_k16}"
TRAINER=TokenModHiCroPL
TEST_BS="${TEST_BS:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRINT_FREQ="${PRINT_FREQ:-20}"
SKIP_DONE="${SKIP_DONE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES

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

RUN_TAG="seed${SEED}_condfilm_tasktext_preservenorm_distill_e${MAX_EPOCH}"
LOG_DIR="output/token_mod/logs"
mkdir -p "${LOG_DIR}"

COMMON_OPTS=(
  DATASET.NUM_SHOTS "${SHOTS}"
  OPTIM.MAX_EPOCH "${MAX_EPOCH}"
  TRAIN.PRINT_FREQ "${PRINT_FREQ}"
  DATALOADER.TEST.BATCH_SIZE "${TEST_BS}"
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
  TRAINER.TOKENMOD.MODULATION film
  TRAINER.TOKENMOD.USE_RESIDUAL True
  TRAINER.TOKENMOD.USE_DISTILL True
  TRAINER.TOKENMOD.RESIDUAL_GATE False
  TRAINER.TOKENMOD.USE_CONDITIONAL_CODES True
  TRAINER.TOKENMOD.VISUAL_CODE_SOURCE task_text
  TRAINER.TOKENMOD.FILM_PRESERVE_NORM True
)

echo "[TokenMod] ROOT=${ROOT_DIR}"
echo "[TokenMod] DATA=${DATA}"
echo "[TokenMod] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[TokenMod] SEED=${SEED} MAX_EPOCH=${MAX_EPOCH} SHOTS=${SHOTS}"
echo "[TokenMod] RUN_TAG=${RUN_TAG}"
echo "[TokenMod] SKIP_DONE=${SKIP_DONE}"
echo "[TokenMod] datasets=${DATASETS[*]}"

has_accuracy() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
  grep -Eq '\* accuracy: [0-9]+(\.[0-9]+)?%' "${log_file}"
}

for DATASET in "${DATASETS[@]}"; do
  TRAIN_DIR="output/token_mod/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${RUN_TAG}"
  TEST_DIR="output/token_mod/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${RUN_TAG}"
  TRAIN_LOG="${LOG_DIR}/${DATASET}_${RUN_TAG}_train.out.txt"
  TEST_LOG="${LOG_DIR}/${DATASET}_${RUN_TAG}_novel.out.txt"

  echo "============================================================"
  echo "[TokenMod] DATASET=${DATASET}"
  echo "  train_dir: ${TRAIN_DIR}"
  echo "  test_dir : ${TEST_DIR}"
  echo "============================================================"

  NEED_TRAIN=1
  if [[ "${SKIP_DONE}" == "1" ]] && [[ -f "${TRAIN_DIR}/VLPromptLearner/model-best.pth.tar" ]] && has_accuracy "${TRAIN_LOG}"; then
    echo "[TokenMod] SKIP TRAIN ${DATASET} (found model-best + accuracy log)"
    NEED_TRAIN=0
  fi

  if [[ "${NEED_TRAIN}" == "1" ]]; then
    echo "[TokenMod] TRAIN ${DATASET}"
    echo "  log: ${TRAIN_LOG}"
    python train.py \
      --root "${DATA}" \
      --seed "${SEED}" \
      --trainer "${TRAINER}" \
      --dataset-config-file "configs/datasets/${DATASET}.yaml" \
      --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
      --output-dir "${TRAIN_DIR}" \
      DATASET.SUBSAMPLE_CLASSES base \
      "${COMMON_OPTS[@]}" \
      2>&1 | tee "${TRAIN_LOG}"
  fi

  NEED_TEST=1
  if [[ "${SKIP_DONE}" == "1" ]] && has_accuracy "${TEST_LOG}"; then
    echo "[TokenMod] SKIP NOVEL ${DATASET} (found accuracy log)"
    NEED_TEST=0
  fi

  if [[ "${NEED_TEST}" == "1" ]]; then
    echo "[TokenMod] NOVEL ${DATASET}"
    echo "  model: ${TRAIN_DIR}"
    echo "  log  : ${TEST_LOG}"
    python train.py \
      --root "${DATA}" \
      --seed "${SEED}" \
      --trainer "${TRAINER}" \
      --dataset-config-file "configs/datasets/${DATASET}.yaml" \
      --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
      --output-dir "${TEST_DIR}" \
      --model-dir "${TRAIN_DIR}" \
      --eval-only \
      DATASET.SUBSAMPLE_CLASSES new \
      "${COMMON_OPTS[@]}" \
      2>&1 | tee "${TEST_LOG}"
  fi
done

echo "[TokenMod] All 11 datasets finished for ${RUN_TAG}."
echo "[TokenMod] Summarize with:"
echo "  python scripts/token_mod/summarize_condfilm_11datasets.py --run-tag ${RUN_TAG}"
