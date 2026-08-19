#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-all}"
RQ_VAE_GPU="${RQ_VAE_GPU:-0}"
RQ_KMEANS_GPU="${RQ_KMEANS_GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-comparison}"
REPORTS_ROOT="${REPORTS_ROOT:-reports/poster_pair_alignment_matrix}"

run_experiment() {
  local gpu="$1"
  local label="$2"
  local config="$3"
  local run_name="$4"

  echo "Starting ${label} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SID_REPORTS_DIR="${REPORTS_ROOT}/${label}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py "${config}" \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

run_original_rqvae() {
  run_experiment \
    "${RQ_VAE_GPU}" \
    "original-rqvae" \
    "experiments/rqvae_poster_full.py" \
    "poster-rqvae-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148"
}

run_aligned_rqvae() {
  run_experiment \
    "${RQ_VAE_GPU}" \
    "pair-aligned-rqvae" \
    "experiments/collaborative_rqvae_poster_full.py" \
    "poster-onerec-pair-rqvae-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148"
}

run_original_rqkmeans() {
  run_experiment \
    "${RQ_KMEANS_GPU}" \
    "original-rqkmeans" \
    "experiments/poster_full_33148_bridge.py" \
    "poster-rqkmeans-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148"
}

run_aligned_rqkmeans() {
  run_experiment \
    "${RQ_KMEANS_GPU}" \
    "pair-aligned-rqkmeans" \
    "experiments/collaborative_rqkmeans_poster_full.py" \
    "poster-onerec-pair-alignment-l3-c256-full-seed42-catalog33148"
}

preflight() {
  CUDA_VISIBLE_DEVICES="${RQ_VAE_GPU},${RQ_KMEANS_GPU}" \
  "${PYTHON_BIN}" - <<'PY'
import torch
from sids_rank_vs_generate.datasets import resolve_dataset
from sids_rank_vs_generate.sids import (
    CollaborativeAwareRQKMeansPipeline,
    CollaborativeAwareRQVAEPipeline,
)

if torch.cuda.device_count() != 2:
    raise SystemExit(
        "Expected exactly two selected GPUs, received "
        f"{torch.cuda.device_count()}"
    )
print(f"RQ-VAE lane GPU: {torch.cuda.get_device_name(0)}")
print(f"RQ-KMeans lane GPU: {torch.cuda.get_device_name(1)}")
files = resolve_dataset("50m")
print(f"Prepared dataset: {files.interactions_path}")
print(f"Prepared embeddings: {files.embeddings_path}")
print("Four-way pair-alignment imports OK")
PY
}

run_rqvae_lane() {
  run_original_rqvae
  run_aligned_rqvae
}

run_rqkmeans_lane() {
  run_original_rqkmeans
  run_aligned_rqkmeans
}

preflight

case "${RUN_SET}" in
  all)
    run_rqvae_lane &
    rqvae_pid=$!
    run_rqkmeans_lane &
    rqkmeans_pid=$!
    task_exit=0
    wait "${rqvae_pid}" || task_exit=$?
    wait "${rqkmeans_pid}" || task_exit=$?
    exit "${task_exit}"
    ;;
  rqvae)
    run_rqvae_lane
    ;;
  rqkmeans)
    run_rqkmeans_lane
    ;;
  original-rqvae)
    run_original_rqvae
    ;;
  aligned-rqvae)
    run_aligned_rqvae
    ;;
  original-rqkmeans)
    run_original_rqkmeans
    ;;
  aligned-rqkmeans)
    run_aligned_rqkmeans
    ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}" >&2
    exit 2
    ;;
esac
