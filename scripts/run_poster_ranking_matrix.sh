#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-ranking_none_all}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-comparison}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/poster_cache}"

run_one() {
  local model="$1"
  local init="$2"
  local assignment="$3"
  local levels="$4"
  local codebook="$5"
  local run_name="poster-ranking-${model}-${init}-${assignment}-none-l${levels}-c${codebook}-full-seed42"

  echo "Starting ${run_name} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SID_MODEL="${model}" \
  R3_INIT="${init}" \
  ASSIGNMENT_METHOD="${assignment}" \
  COLLISION_RESOLUTION="none" \
  EVALUATION_SCOPE="ranking" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
  "${PYTHON_BIN}" run.py experiments/poster_sid_config.py \
    --skip-generation \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

phase_a_none() {
  run_one rqkmeans rqkmeans nearest 3 256
  run_one rqvae rqkmeans nearest 3 256
  run_one rqvae rqkmeans rrs 3 256
  run_one rqvae rqkmeans ecm 3 256
  run_one r3vae spherical nearest 3 256
  run_one r3vae rqkmeans nearest 3 256
}

phase_b_level() {
  local levels="$1"
  run_one rqkmeans rqkmeans nearest "${levels}" 256
  run_one rqvae rqkmeans nearest "${levels}" 256
  run_one r3vae spherical nearest "${levels}" 256
  run_one r3vae rqkmeans nearest "${levels}" 256
}

phase_b_none() {
  phase_b_level 2
  phase_b_level 4
}

phase_c_none() {
  run_one rqkmeans rqkmeans nearest 3 128
  run_one rqvae rqkmeans nearest 3 128
  run_one r3vae spherical nearest 3 128
  run_one r3vae rqkmeans nearest 3 128
}

phase_d_none() {
  run_one rqkmeans rqkmeans nearest 3 512
  run_one rqvae rqkmeans nearest 3 512
  run_one r3vae spherical nearest 3 512
  run_one r3vae rqkmeans nearest 3 512
}

phase_e_assignments() {
  local codebook="$1"
  run_one rqvae rqkmeans rrs 3 "${codebook}"
  run_one rqvae rqkmeans ecm 3 "${codebook}"
}

phase_e_none() {
  phase_e_assignments 128
  phase_e_assignments 512
  run_one rqvae rqkmeans nearest 3 1024
  run_one r3vae spherical nearest 3 1024
}

ranking_none_all() {
  phase_a_none
  phase_b_none
  phase_c_none
  phase_d_none
  phase_e_none
}

case "${RUN_SET}" in
  phase_a_none) phase_a_none ;;
  phase_b_none) phase_b_none ;;
  phase_b_l2_none) phase_b_level 2 ;;
  phase_b_l4_none) phase_b_level 4 ;;
  phase_c_none) phase_c_none ;;
  phase_d_none) phase_d_none ;;
  phase_e_none) phase_e_none ;;
  phase_e_assignments_c128_none) phase_e_assignments 128 ;;
  phase_e_assignments_c512_none) phase_e_assignments 512 ;;
  ranking_none_all) ranking_none_all ;;
  *)
    echo "Unknown RUN_SET: ${RUN_SET}" >&2
    echo "Use phase_a_none, phase_b_none, phase_c_none, phase_d_none," >&2
    echo "phase_e_none, a documented shard, or ranking_none_all" >&2
    exit 2
    ;;
esac
