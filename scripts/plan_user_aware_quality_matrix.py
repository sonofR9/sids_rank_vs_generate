#!/usr/bin/env python3
"""Dry-run матрицы user-aware экспериментов"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sids_rank_vs_generate.user_aware_quality_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
