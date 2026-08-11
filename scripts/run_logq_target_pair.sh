#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-pair}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-comparison}"

run_one() {
  local mode="$1"
  local mode_label="${mode}"
  if [[ "${mode}" == "none" ]]; then
    mode_label="baseline"
  fi
  local run_name="logq-target-rqvae-rqkmeans-nearest-token-l2-c256-aux0p1-${mode_label}-full-seed42"

  echo "Starting ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  LOGQ_MODE="${mode}" \
  "${PYTHON_BIN}" run.py experiments/logq_target_rqvae_token.py \
    --skip-ranking \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

case "${RUN_SET}" in
  baseline) run_one none ;;
  corrected) run_one corrected ;;
  pair)
    run_one none
    run_one corrected
    ;;
  *)
    echo "Unknown RUN_SET: ${RUN_SET}" >&2
    echo "Use baseline, corrected, or pair" >&2
    exit 2
    ;;
esac
