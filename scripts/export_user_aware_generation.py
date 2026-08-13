#!/usr/bin/env python3
"""Export compatible UAQ30 generation runs from MLflow to a normalized CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient


RECALL_KEYS = (
    "evaluations.generation.recall_at_100",
    "generation.recall_at_100",
)
SID_COLUMNS = {
    "icr": ("sid.independent_code_rate",),
    "collision_rate": ("sid.collision_rate",),
    "p95_collision_group_size": ("sid.collision_percentiles.p95",),
    "g_max": ("sid.max_collision",),
    "cur_total": ("sid.codebook_quality.codebook_utilization_rate_total",),
    "code_gini": ("sid.code_gini",),
    "code_entropy": ("sid.code_entropy",),
    "intra_code_similarity": ("sid.intra_code_similarity",),
    "prefix_coherence_1": ("sid.prefix_coherence_at_1",),
    "prefix_coherence_2": ("sid.prefix_coherence_at_2",),
    "prefix_coherence_3": ("sid.prefix_coherence_at_3",),
    "valid_sid_rate": ("sid.valid_sid_rate",),
}
RUN_PATTERN = re.compile(
    r"^uaq30-(?P<version>v[12])-(?P<variant>capacity|anchor)-"
    r"l(?P<num_levels>\d+)-c(?P<codebook_size>\d+)$"
)


def first(metrics: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-id", default="16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    records: list[dict[str, object]] = []
    rejected: list[tuple[str, str]] = []
    for run in client.search_runs([args.experiment_id], max_results=5000):
        name = run.data.tags.get("mlflow.runName", "")
        match = RUN_PATTERN.fullmatch(name)
        if not match:
            continue
        if run.info.status != "FINISHED":
            rejected.append((name, f"status={run.info.status}"))
            continue

        metrics = run.data.metrics
        recall = first(metrics, RECALL_KEYS)
        catalog_size = first(metrics, ("data.catalog_size", "metrics.data.catalog_size"))
        missing = [column for column, keys in SID_COLUMNS.items() if first(metrics, keys) is None]
        if recall is None:
            rejected.append((name, "missing Recall@100"))
            continue
        if catalog_size != 33148:
            rejected.append((name, f"catalog_size={catalog_size}"))
            continue
        if missing:
            rejected.append((name, f"missing SID metrics: {','.join(missing)}"))
            continue

        version = match.group("version")
        variant = match.group("variant")
        record: dict[str, object] = {
            "method": f"UAQ30 {version} {variant}",
            "num_levels": int(match.group("num_levels")),
            "codebook_size": int(match.group("codebook_size")),
            "recall_at_100": recall,
            "run_name": name,
            "run_id": run.info.run_id,
            "catalog_size": int(catalog_size),
        }
        for column, keys in SID_COLUMNS.items():
            record[column] = first(metrics, keys)
        records.append(record)

    columns = [
        "method", "num_levels", "codebook_size", "recall_at_100",
        "run_name", "run_id", "catalog_size", *SID_COLUMNS,
    ]
    frame = pd.DataFrame(records, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["method", "num_levels", "codebook_size"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"UAQ30 generation rows: {len(frame)}")
    if not frame.empty:
        print(frame[["method", "num_levels", "codebook_size", "recall_at_100"]].to_string(index=False))
    for name, reason in rejected:
        print(f"Skipped {name}: {reason}")


if __name__ == "__main__":
    main()
