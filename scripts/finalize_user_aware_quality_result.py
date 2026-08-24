#!/usr/bin/env python3
"""Отметить завершенный matrix run"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sids_rank_vs_generate.user_aware_quality_results import (
    create_completion_marker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--attempt-reports-dir", type=Path, required=True)
    parser.add_argument("--completion-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()

    marker_path, _marker = create_completion_marker(
        reports_root=args.reports_root,
        attempt_reports_dir=args.attempt_reports_dir,
        completion_dir=args.completion_dir,
        run_name=args.run_name,
        attempt_id=args.attempt_id,
        gpu_id=args.gpu_id,
        log_path=args.log_path,
    )
    print(marker_path)


if __name__ == "__main__":
    main()
