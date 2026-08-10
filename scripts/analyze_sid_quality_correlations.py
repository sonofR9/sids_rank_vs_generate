#!/usr/bin/env python3
"""Join SID-only MLflow runs to Recall@100 runs and draw paper-style plots."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr


CONFIGURATION_COLUMNS = (
    "model",
    "init",
    "assignment",
    "collision_resolution",
    "codebook_size",
    "num_levels",
    "seed",
    "dataset",
    "data_scope",
)


@dataclass(frozen=True)
class MetricSpec:
    logged_name: str
    label: str
    file_stem: str


TARGET_METRICS = (
    MetricSpec("sid.independent_code_rate", "ICR", "icr"),
    MetricSpec("sid.collision_rate", "Collision Rate", "collision_rate"),
    MetricSpec(
        "sid.collision_percentiles.p95",
        "p95 Collision Group Size",
        "p95_collision_group_size",
    ),
    MetricSpec("sid.max_collision", "G_max", "g_max"),
    MetricSpec(
        "sid.codebook_quality.codebook_utilization_rate_total",
        "CUR_Total",
        "cur_total",
    ),
    MetricSpec("sid.code_gini", "Code Gini", "code_gini"),
    MetricSpec("sid.code_entropy", "Code Entropy", "code_entropy"),
    MetricSpec(
        "sid.intra_code_similarity",
        "Intra-code Similarity",
        "intra_code_similarity",
    ),
    MetricSpec(
        "sid.prefix_coherence_at_1",
        "Prefix Coherence@1",
        "prefix_coherence_at_1",
    ),
    MetricSpec(
        "sid.prefix_coherence_at_2",
        "Prefix Coherence@2",
        "prefix_coherence_at_2",
    ),
    MetricSpec(
        "sid.prefix_coherence_at_3",
        "Prefix Coherence@3",
        "prefix_coherence_at_3",
    ),
    MetricSpec("sid.valid_sid_rate", "Valid SID Rate", "valid_sid_rate"),
)

METHOD_COLORS = {
    "RQKMeans": "#4C78A8",
    "RQVAE": "#F58518",
    "RQVAE (RRS)": "#E45756",
    "RQVAE (ECM)": "#B279A2",
    "R3VAE (spherical)": "#54A24B",
    "R3VAE (rqkmeans)": "#72B7B2",
}
METHOD_MARKERS = {
    "RQVAE (RRS)": "D",
    "RQVAE (ECM)": "o",
}
METHOD_ORDER = (
    "R3VAE (rqkmeans)",
    "R3VAE (spherical)",
    "RQKMeans",
    "RQVAE (ECM)",
    "RQVAE",
    "RQVAE (RRS)",
)


def _value(row: pd.Series, column: str, default=None):
    value = row.get(column, default)
    return default if pd.isna(value) else value


def _parent_runs(runs: pd.DataFrame) -> pd.DataFrame:
    frame = runs.copy()
    parent_column = "tags.mlflow.parentRunId"
    if parent_column in frame:
        frame = frame[frame[parent_column].isna()]
    if "status" in frame:
        frame = frame[frame["status"] == "FINISHED"]
    return frame.copy()


def _tagged_runs(
    runs: pd.DataFrame,
    *,
    name_prefix: str | None = None,
    require_poster: bool = False,
) -> pd.DataFrame:
    frame = _parent_runs(runs)
    if name_prefix is not None:
        run_names = frame["tags.mlflow.runName"].fillna("")
        frame = frame[run_names.str.startswith(name_prefix)]
    if require_poster:
        poster_tags = frame.get(
            "tags.poster", pd.Series(index=frame.index, dtype="object")
        )
        frame = frame[poster_tags.eq("true")]
    for column, expected in (
        ("tags.dataset", "50m"),
        ("tags.data_scope", "full"),
        ("tags.collision_resolution", "none"),
    ):
        if column in frame:
            frame = frame[frame[column].eq(expected)]
    return frame.copy()


def _detect_recall_metric(runs: pd.DataFrame) -> str:
    candidates = []
    for column in runs.columns:
        if not column.startswith("metrics."):
            continue
        normalized = column.lower().replace("@", "_at_")
        is_recall_100 = "recall" in normalized and re.search(
            r"(?:^|_)100(?:$|_)", normalized
        )
        if not is_recall_100:
            continue
        non_null = int(runs[column].notna().sum())
        generation_priority = int("evaluations.generation" in normalized)
        exact_priority = int(normalized.endswith("recall_at_100"))
        candidates.append((generation_priority, exact_priority, non_null, column))
    if not candidates:
        raise ValueError("No logged Recall@100 metric was found")
    candidates.sort(reverse=True)
    return candidates[0][3]


def _configuration_frame(runs: pd.DataFrame, *, experiment_name: str) -> pd.DataFrame:
    records = []
    for _, row in runs.iterrows():
        try:
            codebook_size = int(_value(row, "tags.codebook_size"))
            num_levels = int(_value(row, "tags.num_levels"))
            seed = int(_value(row, "tags.seed", 42))
        except (TypeError, ValueError):
            continue
        record = {
            "model": str(_value(row, "tags.sid_model", "unknown")).lower(),
            "init": str(_value(row, "tags.init", "unknown")).lower(),
            "assignment": str(_value(row, "tags.assignment", "nearest")).lower(),
            "collision_resolution": str(
                _value(row, "tags.collision_resolution", "none")
            ).lower(),
            "codebook_size": codebook_size,
            "num_levels": num_levels,
            "seed": seed,
            "dataset": str(_value(row, "tags.dataset", "unknown")).lower(),
            "data_scope": str(_value(row, "tags.data_scope", "unknown")).lower(),
            "run_id": str(row["run_id"]),
            "run_name": str(
                _value(row, "tags.mlflow.runName", row.get("run_name", ""))
            ),
            "experiment_name": experiment_name,
            "experiment_id": str(row.get("experiment_id", "")),
            "start_time": row.get("start_time"),
            "git_commit": _value(row, "tags.mlflow.source.git.commit"),
            "git_branch": _value(row, "tags.mlflow.source.git.branch"),
        }
        for column in runs.columns:
            if column.startswith("metrics."):
                record[column] = row.get(column)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _latest_per_configuration(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return (
        frame.sort_values("start_time")
        .drop_duplicates(list(CONFIGURATION_COLUMNS), keep="last")
        .reset_index(drop=True)
    )


def _method_name(row: pd.Series) -> str:
    model = row["model"]
    if model == "r3vae":
        return f"R3VAE ({row['init']})"
    if model == "rqvae" and row["assignment"] != "nearest":
        return f"RQVAE ({str(row['assignment']).upper()})"
    return {"rqkmeans": "RQKMeans", "rqvae": "RQVAE"}.get(model, model)


def _join_runs(
    quality: pd.DataFrame,
    recall: pd.DataFrame,
    recall_metric_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality = _latest_per_configuration(quality)
    recall = _latest_per_configuration(recall)
    recall_columns = [
        *CONFIGURATION_COLUMNS,
        "run_id",
        "run_name",
        "experiment_name",
        "experiment_id",
        "git_commit",
        "git_branch",
        recall_metric_column,
    ]
    merged = quality.merge(
        recall[recall_columns],
        on=list(CONFIGURATION_COLUMNS),
        how="inner",
        suffixes=("_quality", "_recall"),
        validate="one_to_one",
    )
    merged = merged.rename(columns={recall_metric_column: "recall_100"})
    merged["method"] = merged.apply(_method_name, axis=1)

    quality_keys = pd.MultiIndex.from_frame(quality[list(CONFIGURATION_COLUMNS)])
    recall_keys = pd.MultiIndex.from_frame(recall[list(CONFIGURATION_COLUMNS)])
    matched_keys = pd.MultiIndex.from_frame(merged[list(CONFIGURATION_COLUMNS)])
    unmatched_quality = quality[~quality_keys.isin(matched_keys)].copy()
    unmatched_recall = recall[~recall_keys.isin(matched_keys)].copy()
    return merged, unmatched_quality, unmatched_recall


def _finite_pairs(frame: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    pair = frame[[metric_column, "recall_100"]].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(pair[metric_column]) & np.isfinite(pair["recall_100"])
    return frame.loc[mask].copy()


def _correlations(frame: pd.DataFrame, metric_column: str) -> tuple[int, float, float]:
    pair = _finite_pairs(frame, metric_column)
    n = len(pair)
    if n < 2:
        return n, math.nan, math.nan
    x = pair[metric_column].astype(float).to_numpy()
    y = pair["recall_100"].astype(float).to_numpy()
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return n, math.nan, math.nan
    return n, float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def _format_correlation(value: float, *, symbol: str) -> str:
    if not math.isfinite(value):
        return f"{symbol} = undefined (constant)"
    return f"{symbol} = {value:.3f}"


def _draw_sweep_lines(axis: plt.Axes, frame: pd.DataFrame, metric_column: str) -> None:
    for method, method_frame in frame.groupby("method", sort=False):
        color = METHOD_COLORS.get(method, "#6B7280")
        for fixed_column, fixed_value, order_column in (
            ("codebook_size", 256, "num_levels"),
            ("num_levels", 3, "codebook_size"),
        ):
            sweep = method_frame[method_frame[fixed_column] == fixed_value]
            sweep = sweep.sort_values(order_column).drop_duplicates(order_column)
            if len(sweep) < 2:
                continue
            axis.plot(
                sweep[metric_column],
                sweep["recall_100"],
                color=color,
                linewidth=1.15,
                alpha=0.72,
                solid_capstyle="round",
                zorder=1,
            )


def _plot_metric(
    frame: pd.DataFrame,
    spec: MetricSpec,
    output_dir: Path,
    *,
    baseline_recall: float | None,
) -> tuple[int, float, float]:
    metric_column = f"metrics.{spec.logged_name}"
    plot_frame = _finite_pairs(frame, metric_column)
    n, pearson, spearman = _correlations(plot_frame, metric_column)
    if n == 0:
        return n, pearson, spearman

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 5.2), facecolor="white")
    axis.set_facecolor("white")
    _draw_sweep_lines(axis, plot_frame, metric_column)

    present_methods = [
        method for method in METHOD_ORDER if method in set(plot_frame["method"])
    ]
    present_methods.extend(sorted(set(plot_frame["method"]) - set(present_methods)))
    for method in present_methods:
        points = plot_frame[plot_frame["method"] == method]
        color = METHOD_COLORS.get(method, "#6B7280")
        axis.scatter(
            points[metric_column],
            points["recall_100"],
            s=58,
            marker=METHOD_MARKERS.get(method, "o"),
            facecolors="white",
            edgecolors=color,
            linewidths=1.65,
            alpha=0.96,
            zorder=3,
            label=method,
        )

    if baseline_recall is not None:
        axis.axhline(
            baseline_recall,
            color="#6B7280",
            linewidth=1.05,
            linestyle=(0, (4, 3)),
            alpha=0.82,
            zorder=0,
        )
        axis.annotate(
            "SASRec baseline",
            xy=(0.02, baseline_recall),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            color="#6B7280",
            fontsize=8.2,
            ha="left",
            va="bottom",
        )

    axis.set_title(f"{spec.label} → Recall@100", pad=10, fontweight="normal")
    axis.set_xlabel(spec.label, labelpad=8)
    axis.set_ylabel("Recall@100", labelpad=8)
    axis.grid(True, color="#D1D5DB", linewidth=0.65, alpha=0.42)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9CA3AF")
    axis.spines["bottom"].set_color("#9CA3AF")
    axis.tick_params(colors="#374151", width=0.8)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7))
    axis.margins(x=0.08, y=0.10)

    stats_text = "\n".join(
        (
            _format_correlation(pearson, symbol="Pearson r"),
            _format_correlation(spearman, symbol="Spearman ρ"),
            f"n = {n}",
        )
    )
    axis.text(
        0.025,
        0.975,
        stats_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color="#374151",
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.5},
        zorder=5,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS.get(method, "#6B7280"),
            linewidth=1.15,
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.45,
            label=method,
        )
        for method in present_methods
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.15,
    )
    figure.tight_layout(rect=(0, 0.115, 1, 1))
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"recall100_vs_{spec.file_stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return n, pearson, spearman


def _discovered_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in sorted(c for c in frame.columns if c.startswith("metrics.")):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        count = int(np.isfinite(numeric).sum())
        if count:
            rows.append({"metric": column.removeprefix("metrics."), "num_runs": count})
    return pd.DataFrame(rows)


def _write_metric_list(metrics: Iterable[str], path: Path) -> None:
    path.write_text("\n".join(metrics) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--recall-experiment", default="sid-comparison")
    parser.add_argument(
        "--quality-experiment",
        default="sid-quality-only-full-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/poster/sid_quality_correlations"),
    )
    parser.add_argument(
        "--recall-metric",
        help="Exact logged Recall@100 name, with or without the metrics. prefix.",
    )
    parser.add_argument(
        "--baseline-recall",
        type=float,
        help="Optional comparable SASRec Recall@100 horizontal baseline.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    recall_experiment = mlflow.get_experiment_by_name(args.recall_experiment)
    quality_experiment = mlflow.get_experiment_by_name(args.quality_experiment)
    if recall_experiment is None:
        raise SystemExit(f"MLflow experiment not found: {args.recall_experiment}")
    if quality_experiment is None:
        raise SystemExit(f"MLflow experiment not found: {args.quality_experiment}")

    quality_runs = mlflow.search_runs(
        experiment_ids=[quality_experiment.experiment_id],
        output_format="pandas",
    )
    recall_runs = mlflow.search_runs(
        experiment_ids=[recall_experiment.experiment_id],
        output_format="pandas",
    )
    quality_parents = _tagged_runs(quality_runs, name_prefix="sid-quality-")
    recall_parents = _tagged_runs(recall_runs, require_poster=True)

    recall_metric = args.recall_metric
    if recall_metric is None:
        recall_metric_column = _detect_recall_metric(recall_parents)
    else:
        recall_metric_column = recall_metric
        if not recall_metric_column.startswith("metrics."):
            recall_metric_column = f"metrics.{recall_metric_column}"
        if recall_metric_column not in recall_parents:
            raise SystemExit(f"Recall metric not found: {recall_metric_column}")
    recall_parents = recall_parents[recall_parents[recall_metric_column].notna()].copy()

    quality = _configuration_frame(
        quality_parents,
        experiment_name=args.quality_experiment,
    )
    recall = _configuration_frame(
        recall_parents,
        experiment_name=args.recall_experiment,
    )
    merged, unmatched_quality, unmatched_recall = _join_runs(
        quality,
        recall,
        recall_metric_column,
    )
    if merged.empty:
        raise SystemExit("No SID-quality and Recall@100 configurations matched")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovered = _discovered_metrics(quality)
    discovered.to_csv(args.output_dir / "all_sid_metrics.csv", index=False)
    _write_metric_list(discovered["metric"], args.output_dir / "all_sid_metrics.txt")

    quality.to_csv(args.output_dir / "sid_quality_runs.csv", index=False)
    recall.to_csv(args.output_dir / "recall_runs.csv", index=False)
    merged.to_csv(args.output_dir / "matched_runs.csv", index=False)
    unmatched_quality.to_csv(args.output_dir / "unmatched_quality_runs.csv", index=False)
    unmatched_recall.to_csv(args.output_dir / "unmatched_recall_runs.csv", index=False)

    summaries = []
    plotted = []
    for spec in TARGET_METRICS:
        metric_column = f"metrics.{spec.logged_name}"
        if metric_column not in merged or not merged[metric_column].notna().any():
            continue
        n, pearson, spearman = _plot_metric(
            merged,
            spec,
            args.output_dir,
            baseline_recall=args.baseline_recall,
        )
        if n == 0:
            continue
        summaries.append(
            {
                "metric": spec.label,
                "num_runs": n,
                "pearson_corr": pearson,
                "spearman_corr": spearman,
            }
        )
        plotted.append(spec.logged_name)

    summary = pd.DataFrame(summaries)
    summary["_abs_spearman"] = summary["spearman_corr"].abs()
    summary.sort_values(
        ["_abs_spearman", "metric"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )
    summary.drop(columns="_abs_spearman", inplace=True)
    summary.to_csv(args.output_dir / "correlation_summary.csv", index=False)

    metadata = {
        "tracking_uri": args.tracking_uri,
        "quality_experiment": args.quality_experiment,
        "recall_experiment": args.recall_experiment,
        "recall_metric": recall_metric_column.removeprefix("metrics."),
        "quality_parent_runs": len(_latest_per_configuration(quality)),
        "recall_parent_runs": len(_latest_per_configuration(recall)),
        "matched_runs": len(merged),
        "unmatched_quality_runs": len(unmatched_quality),
        "unmatched_recall_runs": len(unmatched_recall),
        "plotted_metrics": plotted,
        "baseline_recall": args.baseline_recall,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
