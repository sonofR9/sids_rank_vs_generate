#!/usr/bin/env python3
"""Check the latest MLflow status of the four pair-alignment matrix runs."""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd


EXPECTED_RUNS = (
    "poster-rqvae-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148",
    "poster-onerec-pair-rqvae-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148",
    "poster-rqkmeans-rqkmeans-nearest-none-l3-c256-full-seed42-catalog33148",
    "poster-onerec-pair-alignment-l3-c256-full-seed42-catalog33148",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default="http://51.250.47.231:5000",
    )
    parser.add_argument("--experiment", default="sid-comparison")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"MLflow experiment not found: {args.experiment}")
    runs = mlflow.search_runs(
        [experiment.experiment_id],
        max_results=10_000,
        output_format="pandas",
    )
    name_column = "tags.mlflow.runName"
    if name_column not in runs:
        raise SystemExit("MLflow result has no run-name tag")

    rows: list[dict[str, str]] = []
    incomplete = False
    failed = False
    for name in EXPECTED_RUNS:
        matching = runs[runs[name_column].eq(name)].sort_values(
            "start_time",
            ascending=False,
        )
        if matching.empty:
            rows.append(
                {"method": name, "status": "MISSING", "run_id": "-"}
            )
            incomplete = True
            continue
        latest = matching.iloc[0]
        status = str(latest["status"])
        rows.append(
            {
                "method": name,
                "status": status,
                "run_id": str(latest["run_id"]),
            }
        )
        if status in {"FAILED", "KILLED"}:
            failed = True
        elif status != "FINISHED":
            incomplete = True

    table = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 100, "display.width", 180):
        print(table.to_string(index=False))
    if failed:
        print("\nAt least one latest run failed or was killed.")
        return 2
    if incomplete:
        print("\nMatrix is not finished yet.")
        return 1
    print("\nAll four runs are FINISHED. Graph export can start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
