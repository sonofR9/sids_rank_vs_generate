#!/usr/bin/env python3
"""Recalculate extended SID metrics and plot them in a copied report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from plot_all_generation_methods import METHOD_ORDER, TARGETS, plot_metric
from sids_rank_vs_generate.evaluation.precomputed_sid_metrics import (
    calculate_all_sid_metrics,
)
from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols


REPOSITORY = Path(__file__).resolve().parents[1]
POINT_KEY = ["method", "num_levels", "codebook_size"]
DEFAULT_SOURCE = Path(
    "reports/figures/all_methods_sid_quality_fixed_transact"
)
DEFAULT_OUTPUT = Path(
    "reports/figures/all_methods_sid_quality_fixed_transact_extended_sid_metrics"
)
DEFAULT_RUNS = Path("reports/transact_sid_token_history/runs")
DEFAULT_EMBEDDINGS = Path(
    "/home/sashanovak/tmp_a100/ysda_recsys/competition/generated/datasets/"
    "yambda/50m_like_core5_knownitems/embeddings_compact.parquet"
)


@dataclass(frozen=True)
class ExtendedMetric:
    column: str
    label: str
    stem: str


def metric_specs() -> tuple[ExtendedMetric, ...]:
    metrics = [
        ExtendedMetric("singleton_rate", "Singleton Item Rate", "singleton_rate"),
        ExtendedMetric(
            "collision_item_rate",
            "Collision Item Rate",
            "collision_item_rate",
        ),
        ExtendedMetric(
            "pair_collision_probability",
            "Pair Collision Probability",
            "pair_collision_probability",
        ),
        ExtendedMetric(
            "intra_code_similarity_v2",
            "Per-level Intra-code Similarity (v2)",
            "intra_code_similarity_v2",
        ),
        ExtendedMetric(
            "prefix_coherence_4",
            "Prefix Coherence@4",
            "prefix_coherence_at_4",
        ),
    ]
    for level in range(1, 5):
        metrics.extend(
            [
                ExtendedMetric(
                    f"adjusted_semantic_variance_explained_{level}",
                    f"Adjusted Semantic Variance Explained@{level}",
                    f"adjusted_semantic_variance_explained_at_{level}",
                ),
                ExtendedMetric(
                    f"normalized_conditional_entropy_{level}",
                    f"Normalized Conditional Entropy@{level}",
                    f"normalized_conditional_entropy_at_{level}",
                ),
                ExtendedMetric(
                    f"pair_refinement_rate_{level}",
                    f"Pair Refinement Rate@{level}",
                    f"pair_refinement_rate_at_{level}",
                ),
                ExtendedMetric(
                    f"mean_active_branch_fraction_{level}",
                    f"Mean Active Branch Fraction@{level}",
                    f"mean_active_branch_fraction_at_{level}",
                ),
                *(
                    ExtendedMetric(
                        f"load_balance_{percentile}_{level}",
                        f"Load Balance {percentile.upper()}@{level}",
                        f"load_balance_{percentile}_at_{level}",
                    )
                    for percentile in ("p25", "p75", "p90")
                ),
                ExtendedMetric(
                    f"codebook_coverage_{level}",
                    f"Codebook Coverage@{level}",
                    f"codebook_coverage_at_{level}",
                ),
                ExtendedMetric(
                    f"codebook_utilization_rate_{level}",
                    f"Codebook Utilization Rate@{level}",
                    f"codebook_utilization_rate_at_{level}",
                ),
            ]
        )
    metrics.extend(
        ExtendedMetric(
            f"load_balance_{percentile}_mean",
            f"Mean Load Balance {percentile.upper()}",
            f"load_balance_{percentile}_mean",
        )
        for percentile in ("p25", "p75", "p90")
    )
    return tuple(metrics)


EXTENDED_METRICS = metric_specs()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def load_run_index(runs_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for metrics_path in runs_root.rglob("metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = payload.get("tracking", {}).get("run_id")
        semantic_ids = metrics_path.with_name("semantic_ids.parquet")
        if run_id and semantic_ids.is_file():
            if run_id in result:
                raise ValueError(f"duplicate local run ID: {run_id}")
            result[str(run_id)] = (semantic_ids, payload)
    return result


def align_embeddings(embeddings: pl.DataFrame, sids: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    id_column = (
        "item_id"
        if "item_id" in embeddings
        else "compact_id" if "compact_id" in embeddings else None
    )
    if id_column is None:
        raise ValueError("embeddings must contain item_id or compact_id")
    embeddings = embeddings.rename({id_column: "item_id"}) if id_column != "item_id" else embeddings
    sid_ids = sids.select("item_id")
    for offset in (0, -1, 1):
        candidate = embeddings.with_columns(
            (pl.col("item_id") + offset).alias("item_id")
        )
        missing = sid_ids.join(candidate.select("item_id"), on="item_id", how="anti")
        if missing.is_empty():
            return candidate, offset
    raise ValueError("could not align compact embedding IDs with SID item IDs")


def codebook_sizes(row: Any, sids: pl.DataFrame) -> tuple[int, ...]:
    levels = len(get_sid_cols(sids))
    size = int(row.codebook_size)
    if row.method == "PLUM":
        sizes = (size * 2, size, size // 2)
    elif row.method == "RQVAE-token":
        sizes = tuple(int(sids[column].max()) + 1 for column in get_sid_cols(sids))
    else:
        sizes = (size,) * levels
    if len(sizes) != levels:
        raise ValueError(
            f"{row.method} L={row.num_levels} has {levels} SID columns but "
            f"{len(sizes)} configured sizes"
        )
    return sizes


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "codebook_quality":
            flat.update(value)
        elif not isinstance(value, dict):
            flat[key] = value
    return flat


def csv_metric_values(flat: dict[str, Any]) -> dict[str, Any]:
    values = {
        "singleton_rate": flat.get("singleton_rate"),
        "collision_item_rate": flat.get("collision_item_rate"),
        "pair_collision_probability": flat.get("pair_collision_probability"),
        "intra_code_similarity_v2": flat.get("intra_code_similarity_v2"),
        "prefix_coherence_4": flat.get("prefix_coherence_at_4"),
    }
    for level in range(1, 5):
        values.update(
            {
                f"adjusted_semantic_variance_explained_{level}": flat.get(
                    f"adjusted_semantic_variance_explained_at_{level}"
                ),
                f"normalized_conditional_entropy_{level}": flat.get(
                    f"normalized_conditional_entropy_at_{level}_layer"
                ),
                f"pair_refinement_rate_{level}": flat.get(
                    f"pair_refinement_rate_at_{level}_layer"
                ),
                f"mean_active_branch_fraction_{level}": flat.get(
                    f"mean_active_branch_fraction_at_{level}_layer"
                ),
                f"codebook_coverage_{level}": flat.get(
                    f"codebook_coverage_at_{level}_layer"
                ),
                f"codebook_utilization_rate_{level}": flat.get(
                    f"codebook_utilization_rate_at_{level}_layer"
                ),
            }
        )
        for percentile in ("p25", "p75", "p90"):
            values[f"load_balance_{percentile}_{level}"] = flat.get(
                f"load_balance_{percentile}_at_{level}_layer"
            )
    for percentile in ("p25", "p75", "p90"):
        values[f"load_balance_{percentile}_mean"] = flat.get(
            f"load_balance_{percentile}_mean"
        )
    return values


def finite_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(np.isclose(float(left), float(right), rtol=1e-9, atol=1e-10))
    except (TypeError, ValueError):
        return left == right


def verify_against_logged(
    calculated: dict[str, Any],
    payload: dict[str, Any],
    run_id: str,
    *,
    allow_size_contract_mismatch: bool,
) -> list[str]:
    logged = flatten_metrics(payload.get("sid", {}).get("statistics", {}))
    allowed_prefixes = (
        "normalized_conditional_entropy_",
        "mean_active_branch_fraction_",
        "load_balance_",
        "codebook_coverage_",
        "codebook_utilization_",
    )
    allowed = []
    for key, value in calculated.items():
        if key in logged and not finite_equal(value, logged[key]):
            if allow_size_contract_mismatch and key.startswith(allowed_prefixes):
                allowed.append(key)
                continue
            raise ValueError(
                f"recalculated metric mismatch for run {run_id}: {key}: "
                f"{value!r} != {logged[key]!r}"
            )
    return allowed


def calculate_table(
    source_rows: pd.DataFrame,
    run_index: dict[str, tuple[Path, dict[str, Any]]],
    embeddings_path: Path,
) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    embeddings_raw = pl.read_parquet(embeddings_path)
    first_run = str(source_rows.iloc[0]["run_id"])
    first_sids = pl.read_parquet(run_index[first_run][0])
    embeddings, id_offset = align_embeddings(embeddings_raw, first_sids)
    cache: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    rows = []
    legacy_mismatches = []
    total = len(source_rows)
    for index, row in enumerate(source_rows.itertuples(index=False), start=1):
        run_id = str(row.run_id)
        if run_id not in run_index:
            raise FileNotFoundError(f"no local semantic_ids.parquet for run {run_id}")
        sid_path, payload = run_index[run_id]
        digest = hashlib.sha256(sid_path.read_bytes()).hexdigest()
        sids = pl.read_parquet(sid_path)
        sizes = codebook_sizes(row, sids)
        cache_key = (digest, sizes)
        if cache_key not in cache:
            calculated = flatten_metrics(
                calculate_all_sid_metrics(
                    sids,
                    embeddings,
                    codebook_sizes=sizes,
                    source=str(sid_path),
                )
            )
            cache[cache_key] = calculated
        calculated = cache[cache_key]
        allowed_mismatches = verify_against_logged(
            calculated,
            payload,
            run_id,
            allow_size_contract_mismatch=row.method == "RQVAE-token",
        )
        if allowed_mismatches:
            legacy_mismatches.append(
                {
                    "run_id": run_id,
                    "method": row.method,
                    "metrics": allowed_mismatches,
                    "reason": (
                        "historical collision-token run mixed nominal and "
                        "observed final-codebook size conventions"
                    ),
                }
            )
        rows.append(
            {
                **{column: getattr(row, column) for column in POINT_KEY},
                "run_id": run_id,
                "semantic_ids_parquet": str(sid_path.relative_to(REPOSITORY)),
                "semantic_ids_sha256": digest,
                "configured_codebook_sizes": "x".join(map(str, sizes)),
                **csv_metric_values(calculated),
            }
        )
        print(
            f"METRICS {index}/{total} method={row.method} "
            f"L={row.num_levels} C={row.codebook_size} unique={len(cache)}",
            flush=True,
        )
    return pd.DataFrame(rows), id_offset, legacy_mismatches


def enrich(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [spec.column for spec in EXTENDED_METRICS]
    result = frame.drop(columns=[column for column in metric_columns if column in frame])
    return result.merge(
        metrics[[*POINT_KEY, *metric_columns]],
        on=POINT_KEY,
        how="left",
        validate="one_to_one",
    )


def update_coverage(
    existing: pd.DataFrame,
    generation: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    new_rows = []
    for target in TARGETS:
        frame = generation if target.key == "recall100" else ranking
        for metric in EXTENDED_METRICS:
            for method in METHOD_ORDER:
                points = frame[frame["method"].eq(method)].dropna(
                    subset=[metric.column, target.column]
                )
                new_rows.append(
                    {
                        "target": target.key,
                        "sid_metric": metric.column,
                        "method": method,
                        "num_points": len(points),
                    }
                )
    return pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    args = parser.parse_args()

    source_dir = resolve(args.source_dir)
    output_dir = resolve(args.output_dir)
    runs_root = resolve(args.runs_root)
    embeddings_path = resolve(args.embeddings)
    for required in (
        source_dir / "generation_points.csv",
        source_dir / "ranking_points.csv",
        source_dir / "metric_coverage.csv",
        source_dir / "transact_metrics.csv",
        embeddings_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    shutil.copytree(
        source_dir,
        output_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("README.md"),
    )
    source_metrics = pd.read_csv(source_dir / "transact_metrics.csv")
    if source_metrics.duplicated(POINT_KEY).any():
        raise ValueError("transact_metrics.csv contains duplicate variant keys")
    calculated, id_offset, legacy_mismatches = calculate_table(
        source_metrics,
        load_run_index(runs_root),
        embeddings_path,
    )
    calculated.to_csv(output_dir / "sid_metrics_calculated.csv", index=False)

    generation = enrich(
        pd.read_csv(source_dir / "generation_points.csv"), calculated
    )
    ranking = enrich(pd.read_csv(source_dir / "ranking_points.csv"), calculated)
    transact_metrics = enrich(source_metrics, calculated)
    generation.to_csv(output_dir / "generation_points.csv", index=False)
    ranking.to_csv(output_dir / "ranking_points.csv", index=False)
    transact_metrics.to_csv(output_dir / "transact_metrics.csv", index=False)
    update_coverage(
        pd.read_csv(source_dir / "metric_coverage.csv"), generation, ranking
    ).to_csv(output_dir / "metric_coverage.csv", index=False)

    plotted = 0
    for target in TARGETS:
        frame = generation if target.key == "recall100" else ranking
        for metric in EXTENDED_METRICS:
            if plot_metric(
                frame,
                metric,
                target,
                output_dir / "extended_sid_metrics" / target.key,
            ):
                plotted += 1

    metadata = {
        "source_report": str(source_dir.relative_to(REPOSITORY)),
        "embeddings": str(embeddings_path),
        "embedding_id_offset": id_offset,
        "runs_root": str(runs_root.relative_to(REPOSITORY)),
        "variant_rows": len(calculated),
        "unique_sid_parquets": int(calculated["semantic_ids_sha256"].nunique()),
        "extended_metric_count": len(EXTENDED_METRICS),
        "extended_metrics": [metric.column for metric in EXTENDED_METRICS],
        "plots_per_format": plotted,
        "legacy_logged_metric_mismatches": legacy_mismatches,
    }
    (output_dir / "extended_sid_metrics_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"COMPLETE variants={len(calculated)} unique_sids="
        f"{calculated['semantic_ids_sha256'].nunique()} "
        f"metrics={len(EXTENDED_METRICS)} plots_per_format={plotted} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
