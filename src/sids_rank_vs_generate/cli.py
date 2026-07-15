from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run shared SIDS interaction preprocessing from scratch."
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=Path("interactions.parquet"),
        help="Path to interactions.parquet (default: ./interactions.parquet).",
    )
    parser.add_argument(
        "--core-min-interactions",
        type=int,
        default=5,
        help="Minimum interactions required for an item (default: 5).",
    )
    parser.add_argument(
        "--test-days",
        type=float,
        default=7.0,
        help="Length of the chronological holdout in days (default: 7).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path for a machine-readable preprocessing summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=args.interactions,
            core_min_interactions_per_item=args.core_min_interactions,
            test_days=args.test_days,
            seed=args.seed,
        )
    )
    summary = prepared.summary()
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered + "\n", encoding="utf-8")
    return 0
