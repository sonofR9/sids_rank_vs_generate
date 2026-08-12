#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-pair}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-logq-generation}"
SID_LOGQ_BETA="${SID_LOGQ_BETA:-1.0}"

run_one() {
  local mode="$1"
  local label="${mode}"
  if [[ "${mode}" == "none" ]]; then
    label="baseline"
  else
    label="logq"
  fi
  local beta_label="${SID_LOGQ_BETA//./p}"
  local run_name="sid-logq-rqvae-rqkmeans-nearest-token-l2-c256-aux0-${label}-beta${beta_label}-full-seed42"

  echo "Starting ${run_name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SID_LOGQ_MODE="${mode}" \
  SID_LOGQ_BETA="${SID_LOGQ_BETA}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/sid_logq_rqvae.py \
    --skip-ranking \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

case "${RUN_SET}" in
  baseline) run_one none ;;
  logq) run_one standard ;;
  pair)
    run_one none
    run_one standard
    ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}; use baseline, logq, or pair" >&2
    exit 2
    ;;
esac
