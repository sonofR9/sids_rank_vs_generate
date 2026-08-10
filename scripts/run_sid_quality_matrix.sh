#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-smoke}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_SIZE="${DATASET_SIZE:-50m}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-quality-only}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/poster_cache}"

run_one() {
  local model="$1"
  local init="$2"
  local assignment="$3"
  local levels="$4"
  local codebook="$5"
  local run_name="sid-quality-${model}-${init}-${assignment}-l${levels}-c${codebook}-full-seed42"

  echo "Starting ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SID_MODEL="${model}" \
  R3_INIT="${init}" \
  ASSIGNMENT_METHOD="${assignment}" \
  COLLISION_RESOLUTION="none" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  DATASET_SIZE="${DATASET_SIZE}" \
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
  "${PYTHON_BIN}" run.py experiments/poster_sid_config.py \
    --skip-ranking \
    --skip-generation \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

anchors() {
  run_one rqkmeans rqkmeans nearest 3 256
  run_one rqvae rqkmeans nearest 3 256
  run_one rqvae rqkmeans rrs 3 256
  run_one rqvae rqkmeans ecm 3 256
  run_one r3vae spherical nearest 3 256
  run_one r3vae rqkmeans nearest 3 256
}

levels() {
  local level
  for level in 2 4; do
    run_one rqkmeans rqkmeans nearest "${level}" 256
    run_one rqvae rqkmeans nearest "${level}" 256
    run_one r3vae spherical nearest "${level}" 256
    run_one r3vae rqkmeans nearest "${level}" 256
  done
}

sizes() {
  local codebook
  for codebook in 128 512; do
    run_one rqkmeans rqkmeans nearest 3 "${codebook}"
    run_one rqvae rqkmeans nearest 3 "${codebook}"
    run_one rqvae rqkmeans rrs 3 "${codebook}"
    run_one rqvae rqkmeans ecm 3 "${codebook}"
    run_one r3vae spherical nearest 3 "${codebook}"
    run_one r3vae rqkmeans nearest 3 "${codebook}"
  done
}

extras() {
  run_one rqvae rqkmeans nearest 3 1024
  run_one r3vae spherical nearest 3 1024
}

case "${RUN_SET}" in
  smoke) run_one rqkmeans rqkmeans nearest 3 256 ;;
  anchors) anchors ;;
  levels) levels ;;
  sizes) sizes ;;
  extras) extras ;;
  all)
    anchors
    levels
    sizes
    extras
    ;;
  *)
    echo "Unknown RUN_SET: ${RUN_SET}" >&2
    echo "Use smoke, anchors, levels, sizes, extras, or all" >&2
    exit 2
    ;;
esac
