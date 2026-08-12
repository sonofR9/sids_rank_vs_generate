"""Один запуск из зафиксированной матрицы user-aware SID"""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.user_aware_quality_experiment import (
    build_user_aware_quality_experiment_by_name,
)


RUN_NAME = os.environ.get("SID_MATRIX_RUN", "").strip()
if not RUN_NAME:
    raise RuntimeError(
        "set SID_MATRIX_RUN to a run_name from "
        "scripts/plan_user_aware_quality_matrix.py"
    )

REPORTS_DIR = Path(
    os.environ.get("SID_REPORTS_DIR", "reports/experiments")
)

EXPERIMENT = build_user_aware_quality_experiment_by_name(
    RUN_NAME,
    reports_dir=REPORTS_DIR,
)
