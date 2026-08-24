#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${SID_MAPPING_LOG_DIR:-${ROOT_DIR}/reports/sid_mappings_logs}"
mkdir -p "${LOG_DIR}"

run_worker() {
  local gpu="$1"
  local label="$2"
  shift 2
  (
    cd "${ROOT_DIR}"
    echo "[$(date -u +%FT%H:%M:%SZ)] GPU ${gpu}: starting $*"
    for run_set in "$@"; do
      echo "[$(date -u +%FT%H:%M:%SZ)] GPU ${gpu}: RUN_SET=${run_set}"
      GPU="${gpu}" RUN_SET="${run_set}" \
        bash scripts/export_sid_mapping_matrix.sh
    done
    echo "[$(date -u +%FT%H:%M:%SZ)] GPU ${gpu}: completed"
  ) >"${LOG_DIR}/sid-mappings-${label}.log" 2>&1 &
  RUN_PID=$!
}

run_worker 0 gpu0 rqvae balance
GPU0_PID="${RUN_PID}"
run_worker 1 gpu1 rqkmeans r3vae plum
GPU1_PID="${RUN_PID}"

STATUS=0
if ! wait "${GPU0_PID}"; then
  echo "GPU 0 worker failed; inspect ${LOG_DIR}/sid-mappings-gpu0.log" >&2
  STATUS=1
fi
if ! wait "${GPU1_PID}"; then
  echo "GPU 1 worker failed; inspect ${LOG_DIR}/sid-mappings-gpu1.log" >&2
  STATUS=1
fi
exit "${STATUS}"
