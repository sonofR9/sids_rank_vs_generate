#!/usr/bin/env bash
set -euo pipefail

RUN_SET="${RUN_SET:-all}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://51.250.47.231:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-sid-comparison}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-reports/poster_cache}"

run_one() {
  local model="$1"
  local init="$2"
  local assignment="$3"
  local collision="$4"
  local levels="$5"
  local codebook="$6"
  local run_name="poster-${model}-${init}-${assignment}-${collision}-l${levels}-c${codebook}-full-seed42"

  echo "Starting ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  SID_MODEL="${model}" \
  R3_INIT="${init}" \
  ASSIGNMENT_METHOD="${assignment}" \
  COLLISION_RESOLUTION="${collision}" \
  NUM_LEVELS="${levels}" \
  CODEBOOK_SIZE="${codebook}" \
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
  "${PYTHON_BIN}" run.py experiments/poster_sid_config.py \
    --skip-ranking \
    --mlflow \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --mlflow-experiment-name "${MLFLOW_EXPERIMENT_NAME}" \
    --mlflow-run-name "${run_name}"
}

phase_a() {
  # Anchor and collision-resolution ablations at L=3, C=256.
  # Existing token runs for RQKMeans and both R3VAE initializations are reused.
  run_one rqkmeans rqkmeans nearest none 3 256
  run_one rqvae rqkmeans nearest none 3 256
  run_one rqvae rqkmeans nearest token 3 256
  run_one rqvae rqkmeans rrs none 3 256
  run_one rqvae rqkmeans ecm none 3 256
  run_one r3vae spherical nearest none 3 256
  run_one r3vae rqkmeans nearest none 3 256
}

phase_b_level() {
  # One number-of-levels endpoint, with raw SIDs and no collision token.
  local levels="$1"
  run_one rqkmeans rqkmeans nearest none "${levels}" 256
  run_one rqvae rqkmeans nearest none "${levels}" 256
  run_one r3vae spherical nearest none "${levels}" 256
  run_one r3vae rqkmeans nearest none "${levels}" 256
}

phase_b() {
  # Both number-of-levels endpoints, run sequentially on one GPU.
  phase_b_level 2
  phase_b_level 4
}

phase_c() {
  # Codebook-size control at the central number of levels.
  run_one rqkmeans rqkmeans nearest none 3 128
  run_one rqvae rqkmeans nearest none 3 128
  run_one r3vae spherical nearest none 3 128
  run_one r3vae rqkmeans nearest none 3 128
}

phase_d() {
  # C=512 extension of the codebook-size sweep at L=3.
  run_one rqkmeans rqkmeans nearest none 3 512
  run_one rqvae rqkmeans nearest none 3 512
  run_one r3vae spherical nearest none 3 512
  run_one r3vae rqkmeans nearest none 3 512
}

phase_e_assignments() {
  local codebook="$1"
  run_one rqvae rqkmeans rrs none 3 "${codebook}"
  run_one rqvae rqkmeans ecm none 3 "${codebook}"
}

phase_e_extras() {
  run_one rqvae rqkmeans nearest none 3 1024
  run_one rqvae rqkmeans nearest token 2 256
  run_one r3vae spherical nearest none 3 1024
}

phase_e() {
  # Complete assignment sweeps and selected C=1024/level extensions.
  phase_e_assignments 128
  phase_e_assignments 512
  phase_e_extras
}

case "${RUN_SET}" in
  phase_a) phase_a ;;
  phase_b) phase_b ;;
  phase_b_l2) phase_b_level 2 ;;
  phase_b_l4) phase_b_level 4 ;;
  phase_c) phase_c ;;
  phase_d) phase_d ;;
  phase_d_rqkmeans) run_one rqkmeans rqkmeans nearest none 3 512 ;;
  phase_d_rqvae) run_one rqvae rqkmeans nearest none 3 512 ;;
  phase_d_r3_spherical) run_one r3vae spherical nearest none 3 512 ;;
  phase_d_r3_rqkmeans) run_one r3vae rqkmeans nearest none 3 512 ;;
  phase_e) phase_e ;;
  phase_e_assignments_c128) phase_e_assignments 128 ;;
  phase_e_assignments_c512) phase_e_assignments 512 ;;
  phase_e_extras) phase_e_extras ;;
  phase_e_r3_spherical_c1024) run_one r3vae spherical nearest none 3 1024 ;;
  phase_e_rqvae_c1024) run_one rqvae rqkmeans nearest none 3 1024 ;;
  phase_e_rqvae_token_l2_c256) run_one rqvae rqkmeans nearest token 2 256 ;;
  all)
    phase_a
    phase_b
    phase_c
    phase_d
    phase_e
    ;;
  *)
    echo "Unknown RUN_SET: ${RUN_SET}" >&2
    echo "Use phase_a, phase_b, phase_c, phase_d, a documented shard, or all" >&2
    exit 2
    ;;
esac
