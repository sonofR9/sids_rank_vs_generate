from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from .pipeline import PipelineConfig, run_pipeline
from .sid_experiment import (
    SidExperimentConfig,
    run_sid_experiment,
    write_sid_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run shared preprocessing or a pluggable SID evaluation experiment."
        )
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
    parser.add_argument(
        "--sid",
        help=(
            "Optional SID plugin as module:attribute or path.py:attribute. "
            "The attribute must be an ISidPipeline class, factory, or instance."
        ),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Item-feature parquet required when --sid is supplied.",
    )
    parser.add_argument(
        "--ranking-evaluator",
        help="Optional ranking IEvaluationStage plugin.",
    )
    parser.add_argument(
        "--candgen-evaluator",
        help="Optional candidate-generation IEvaluationStage plugin.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional path for complete SID experiment metrics.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        help="Optional path for a concise SID experiment Markdown report.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Default SID artifact directory (default: reports/experiments).",
    )
    parser.add_argument(
        "--run-name",
        help="Artifact basename for a SID run (default: plugin attribute name).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sid is not None:
        if args.embeddings is None:
            parser.error("--embeddings is required when --sid is supplied")
        evaluator_specs = {
            name: spec
            for name, spec in {
                "ranking": args.ranking_evaluator,
                "candgen": args.candgen_evaluator,
            }.items()
            if spec is not None
        }
        result = run_sid_experiment(
            SidExperimentConfig(
                interactions_path=args.interactions,
                embeddings_path=args.embeddings,
                sid_spec=args.sid,
                evaluator_specs=evaluator_specs,
                core_min_interactions_per_item=args.core_min_interactions,
                test_days=args.test_days,
                seed=args.seed,
            )
        )
        default_name = args.sid.rpartition(":")[2]
        requested_name = args.run_name or default_name
        run_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", requested_name).strip("._")
        if not run_name:
            parser.error("--run-name must contain at least one filename character")
        reports_dir = args.reports_dir or Path("reports/experiments")
        metrics_json = args.metrics_json or reports_dir / f"{run_name}.json"
        report_md = args.report_md or reports_dir / f"{run_name}.md"
        result["artifacts"] = {
            "metrics_json": str(metrics_json),
            "report_md": str(report_md),
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print(rendered)
        write_sid_artifacts(
            result,
            metrics_json=metrics_json,
            report_md=report_md,
        )
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(result["data"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0

    sid_only_options = {
        "--embeddings": args.embeddings,
        "--ranking-evaluator": args.ranking_evaluator,
        "--candgen-evaluator": args.candgen_evaluator,
        "--metrics-json": args.metrics_json,
        "--report-md": args.report_md,
        "--reports-dir": args.reports_dir,
        "--run-name": args.run_name,
    }
    supplied_without_sid = [
        option for option, value in sid_only_options.items() if value is not None
    ]
    if supplied_without_sid:
        parser.error(f"{', '.join(supplied_without_sid)} require --sid")

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
