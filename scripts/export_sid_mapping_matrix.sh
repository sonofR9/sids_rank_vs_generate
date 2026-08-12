#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-mappings-full}"
REPORTS_DIR="${SID_MAPPING_REPORTS_DIR:-reports/sid_mappings}"
RUN_SET="${RUN_SET:-all}"
INTERACTIONS_PATH="${SID_INTERACTIONS_PATH:-}"
EMBEDDINGS_PATH="${SID_EMBEDDINGS_PATH:-}"

run_one() {
  local method="$1"
  local levels="$2"
  local codebook="$3"
  local assignment="${4:-nearest}"
  local r3vae_init="${5:-rqkmeans}"
  local run_name="sid-mapping-${method}-l${levels}-c${codebook}-${assignment}-seed42"

  if [[ "${method}" == "r3vae" ]]; then
    run_name="${run_name}-${r3vae_init}"
  fi
  echo "Starting ${run_name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SID_METHOD="${method}" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  ASSIGNMENT="${assignment}" \
  R3VAE_INIT="${r3vae_init}" \
  SID_MAPPING_REPORTS_DIR="${REPORTS_DIR}" \
  SID_INTERACTIONS_PATH="${INTERACTIONS_PATH}" \
  SID_EMBEDDINGS_PATH="${EMBEDDINGS_PATH}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" run.py experiments/export_sid_mappings.py \
    --sid-only \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

base() {
  for levels_codebook in "2 256" "3 256" "4 256" "3 128" "3 512" "3 1024"; do
    read -r levels codebook <<<"${levels_codebook}"
    run_one rqkmeans "${levels}" "${codebook}"
    run_one rqvae "${levels}" "${codebook}" nearest
    run_one r3vae "${levels}" "${codebook}" nearest rqkmeans
    run_one r3vae "${levels}" "${codebook}" nearest spherical
    if [[ "${levels}" == "3" && "${codebook}" =~ ^(128|256|512)$ ]]; then
      run_one plum "${levels}" "${codebook}"
    fi
  done
  for codebook in 128 256 512; do
    run_one rqvae 3 "${codebook}" rrs
    run_one rqvae 3 "${codebook}" ecm
    run_one balance 3 "${codebook}"
  done
}

case "${RUN_SET}" in
  rqkmeans) for pair in "2 256" "3 256" "4 256" "3 128" "3 512" "3 1024"; do read -r levels codebook <<<"${pair}"; run_one rqkmeans "${levels}" "${codebook}"; done ;;
  rqvae) for pair in "2 256" "3 256" "4 256" "3 128" "3 512" "3 1024"; do read -r levels codebook <<<"${pair}"; run_one rqvae "${levels}" "${codebook}" nearest; done; for codebook in 128 256 512; do run_one rqvae 3 "${codebook}" rrs; run_one rqvae 3 "${codebook}" ecm; done ;;
  r3vae) for pair in "2 256" "3 256" "4 256" "3 128" "3 512" "3 1024"; do read -r levels codebook <<<"${pair}"; run_one r3vae "${levels}" "${codebook}" nearest rqkmeans; run_one r3vae "${levels}" "${codebook}" nearest spherical; done ;;
  balance) for codebook in 128 256 512; do run_one balance 3 "${codebook}"; done ;;
  all) base ;;
  plum) for codebook in 128 256 512; do run_one plum 3 "${codebook}"; done ;;
  *) echo "Unknown RUN_SET=${RUN_SET}; use rqkmeans, rqvae, r3vae, balance, plum, or all" >&2; exit 2 ;;
esac
