from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .datasets import DATASET_SIZES, DEFAULT_DATASET_SIZE, resolve_dataset
from .sid_experiment import SidExperimentConfig, run_sid_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the common SID evaluation pipeline."
    )
    parser.add_argument(
        "--dataset-size",
        choices=DATASET_SIZES,
        default=DEFAULT_DATASET_SIZE,
        help=f"Yambda scale from the shared cache (default: {DEFAULT_DATASET_SIZE}).",
    )
    parser.add_argument(
        "--sid",
        required=True,
        help=(
            "SID implementation as module:attribute or path.py:attribute. "
            "The attribute must be an ISidPipeline class, factory, or instance."
        ),
    )
    parser.add_argument(
        "--ranking",
        action="store_true",
        help="Run the common ranking evaluation stage.",
    )
    parser.add_argument(
        "--candgen",
        action="store_true",
        help="Run the common candidate-generation evaluation stage.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/experiments"),
        help="Root directory for all generated artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = resolve_dataset(args.dataset_size)
    enabled_stages = tuple(
        name
        for name, enabled in (
            ("ranking", args.ranking),
            ("candgen", args.candgen),
        )
        if enabled
    )
    result = run_sid_experiment(
        SidExperimentConfig(
            dataset=dataset,
            sid_spec=args.sid,
            enabled_stages=enabled_stages,
            reports_dir=args.reports_dir,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
