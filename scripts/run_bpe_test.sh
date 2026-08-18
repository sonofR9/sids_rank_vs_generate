#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-bpe}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/poster_cache}"

RUN_SET="${RUN_SET:-all}"

run_one() {
  local levels="$1"
  local codebook="$2"
  local merges="$3"
  local run_name="bpe-l${levels}-c${codebook}-m${merges}"

  echo "Starting ${run_name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  ASSIGNMENT_METHOD="nearest" \
  COLLISION_RESOLUTION="none" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  NUM_MERGES="${merges}" \
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/bpe_full.py \
      --skip-ranking \
      --mlflow \
      --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
      --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
      --mlflow-run-name "${run_name}"
}

case "${RUN_SET}" in
  l7-c256) run_one 7 256 2048 ;;
  l10-c256) run_one 10 256 2048 ;;
  l7-c128) run_one 7 128 1024 ;;
  l7-c32) run_one 7 32 512 ;;
  l10-c32) run_one 10 32 512 ;;

  all)
    run_one 7 256 2048
    run_one 10 256 2048
    run_one 7 128 1024
    run_one 7 32 512
    run_one 10 32 512
    ;;

  *)
    echo "Unknown RUN_SET=${RUN_SET}" >&2
    exit 2
    ;;
esac