#!/usr/bin/env python3
"""Export all-method SID-quality figures for Recall@100, DCNv2, and TransAct.

The exporter combines committed report tables rather than querying MLflow.  It
writes a two-panel poster figure for every SID metric and each downstream
target. PLUM and BalanceRQKMeans are accepted only when a source row explicitly
identifies the shared 33,148-item catalog, so legacy 12,464-item runs cannot be
mixed into the comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


LEVEL_SWEEP = (2, 3, 4)
CODEBOOK_SWEEP = (128, 256, 512, 1024)


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    stem: str
    source_columns: tuple[str, ...]


@dataclass(frozen=True)
class TargetSpec:
    key: str
    column: str
    label: str
    baseline: float | None = None


SID_METRICS = (
    MetricSpec("icr", "Raw independent code rate (ICR)", "icr", ("icr", "metrics.sid.independent_code_rate", "metrics.analysis.raw_independent_code_rate", "metrics.assignment.base_icr")),
    MetricSpec("collision_rate", "Collision Rate", "collision_rate", ("collision_rate", "metrics.sid.collision_rate")),
    MetricSpec("p95_collision_group_size", "p95 Collision Group Size", "p95_collision_group_size", ("p95_collision_group_size", "metrics.sid.collision_percentiles.p95")),
    MetricSpec("g_max", "G_max", "g_max", ("g_max", "metrics.sid.max_collision")),
    MetricSpec("cur_total", "CUR_Total", "cur_total", ("cur_total", "metrics.sid.codebook_quality.codebook_utilization_rate_total")),
    MetricSpec("code_gini", "Code Gini", "code_gini", ("code_gini", "metrics.sid.code_gini")),
    MetricSpec("code_entropy", "Code Entropy", "code_entropy", ("code_entropy", "metrics.sid.code_entropy")),
    MetricSpec("intra_code_similarity", "Intra-code Similarity", "intra_code_similarity", ("intra_code_similarity", "metrics.sid.intra_code_similarity")),
    MetricSpec("prefix_coherence_1", "Prefix Coherence@1", "prefix_coherence_at_1", ("prefix_coherence_1", "metrics.sid.prefix_coherence_at_1")),
    MetricSpec("prefix_coherence_2", "Prefix Coherence@2", "prefix_coherence_at_2", ("prefix_coherence_2", "metrics.sid.prefix_coherence_at_2")),
    MetricSpec("prefix_coherence_3", "Prefix Coherence@3", "prefix_coherence_at_3", ("prefix_coherence_3", "metrics.sid.prefix_coherence_at_3")),
    MetricSpec("valid_sid_rate", "Valid SID Rate", "valid_sid_rate", ("valid_sid_rate", "metrics.sid.valid_sid_rate")),
)

TARGETS = (
    TargetSpec("recall100", "recall_at_100", "Recall@100", 0.113),
    TargetSpec("dcnv2", "dcnv2_pairwise_accuracy", "DCNv2 Pairwise Accuracy"),
    TargetSpec("transact", "transact_pairwise_accuracy", "TransAct Pairwise Accuracy"),
)

BASE_METHOD_ORDER = (
    "R3VAE (rqkmeans)", "R3VAE (spherical)", "RQKMeans", "RQVAE (ECM)",
    "RQVAE", "RQVAE (RRS)", "RQVAE-token",
)
METHOD_ORDER = (*BASE_METHOD_ORDER, "PLUM", "BalanceRQKMeans", "RQVAE + item LogQ", "RQVAE + pure SID LogQ")
METHOD_STYLE = {
    "R3VAE (rqkmeans)": ("#72B7B2", "o", False), "R3VAE (spherical)": ("#54A24B", "o", False),
    "RQKMeans": ("#4C78A8", "o", False), "RQVAE (ECM)": ("#B279A2", "o", False),
    "RQVAE": ("#F58518", "o", False), "RQVAE (RRS)": ("#E45756", "D", False),
    "RQVAE-token": ("#FF9DA6", "s", True), "PLUM": ("#9D755D", "^", False),
    "BalanceRQKMeans": ("#7A5195", "s", False), "RQVAE + item LogQ": ("#8C6D31", "P", False),
    "RQVAE + pure SID LogQ": ("#E15759", "X", False),
}
EXTENSION_METHODS = {"PLUM", "BalanceRQKMeans"}


def read_csv(path: Path, source: str, *, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise SystemExit(f"Report table not found: {path}")
        return pd.DataFrame({"source": pd.Series(dtype=str)})
    frame = pd.read_csv(path)
    frame["source"] = source
    return frame


def first_existing(*candidates: Path) -> Path:
    """Choose the current report layout, retaining support for older exports."""
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def first_present(frame: pd.DataFrame, *columns: str) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=object)
    for column in columns:
        if column in frame:
            values = values.where(values.notna(), frame[column])
    return values


def normalize(frame: pd.DataFrame, *, method: str | None = None) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["method"] = method if method else first_present(frame, "method")
    output["num_levels"] = first_present(frame, "num_levels")
    output["codebook_size"] = first_present(frame, "codebook_size")
    output["recall_at_100"] = first_present(frame, "recall_at_100", "recall_100")
    output["dcnv2_pairwise_accuracy"] = first_present(frame, "dcnv2_pairwise_accuracy")
    output["transact_pairwise_accuracy"] = first_present(frame, "transact_pairwise_accuracy")
    output["run_name"] = first_present(frame, "run_name", "run_name_recall", "sid_quality_run_name", "ranking_run_name", "run_name_quality")
    output["catalog_size"] = first_present(frame, "catalog_size", "metrics.data.catalog_size")
    output["source"] = frame["source"]
    for spec in SID_METRICS:
        output[spec.column] = first_present(frame, *spec.source_columns)
    return output


def is_33148_extension(frame: pd.DataFrame) -> pd.Series:
    catalog = pd.to_numeric(frame["catalog_size"], errors="coerce").eq(33148)
    named = frame["run_name"].astype(str).str.contains(r"(?:^|-)catalog33148(?:-|$)", regex=True, na=False)
    return catalog | named


def filter_extensions(frame: pd.DataFrame) -> pd.DataFrame:
    is_extension = frame["method"].isin(EXTENSION_METHODS)
    return frame[~is_extension | is_33148_extension(frame)].copy()


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = ["num_levels", "codebook_size", *(target.column for target in TARGETS), *(spec.column for spec in SID_METRICS)]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["method", "num_levels", "codebook_size"])


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_key"] = list(zip(result.method, result.num_levels, result.codebook_size, result.run_name))
    return result.drop_duplicates("_key", keep="last").drop(columns="_key")


def padded_limits(values: Iterable[float]) -> tuple[float, float]:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 1.0
    low, high = float(clean.min()), float(clean.max())
    span = high - low
    padding = max(abs(low) * 0.04, 0.01) if span == 0 else span * 0.08
    return low - padding, high + padding


def draw_panel(ax: plt.Axes, frame: pd.DataFrame, x_column: str, y_column: str, order_column: str, title: str, sweep_text: str, x_limits: tuple[float, float], y_limits: tuple[float, float], baseline: float | None) -> None:
    for method in METHOD_ORDER:
        points = frame[frame.method.eq(method)].sort_values(order_column)
        points = points.drop_duplicates(order_column).dropna(subset=[x_column, y_column])
        if points.empty:
            continue
        color, marker, filled = METHOD_STYLE[method]
        if len(points) > 1:
            ax.plot(points[x_column], points[y_column], color=color, linewidth=1.25, alpha=0.72, zorder=1)
        ax.scatter(points[x_column], points[y_column], s=58, marker=marker, facecolors=color if filled else "white", edgecolors=color, linewidths=1.55, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, color="#7B8794", linestyle=(0, (5, 3)), linewidth=1.2, zorder=0)
    ax.set_title(title, pad=9, fontweight="normal")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.42)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3B8")
    ax.tick_params(colors="#334155", width=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.text(0.5, -0.18, sweep_text, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#64748B")
    ax.text(0.99, 0.985, "better ↗", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#7B8794")


def plot_metric(frame: pd.DataFrame, sid_metric: MetricSpec, target: TargetSpec, output_dir: Path) -> bool:
    left = frame[frame.codebook_size.eq(256) & frame.num_levels.isin(LEVEL_SWEEP)]
    right = frame[frame.num_levels.eq(3) & frame.codebook_size.isin(CODEBOOK_SWEEP)]
    shown = pd.concat([left, right], ignore_index=True).dropna(subset=[sid_metric.column, target.column])
    shown = shown.drop_duplicates(["method", "run_name"])
    if shown.empty:
        return False
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 7.7, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), sharey=True, facecolor="white")
    x_limits = padded_limits(shown[sid_metric.column])
    y_values = shown[target.column] if target.baseline is None else pd.concat([shown[target.column], pd.Series([target.baseline])])
    y_limits = padded_limits(y_values)
    draw_panel(axes[0], left, sid_metric.column, target.column, "num_levels", "Varying number of codebooks (C=256)", "L: 2 → 3 → 4", x_limits, y_limits, target.baseline)
    draw_panel(axes[1], right, sid_metric.column, target.column, "codebook_size", "Varying codebook size (L=3)", "C: 128 → 256 → 512 → 1024", x_limits, y_limits, target.baseline)
    figure.supxlabel(sid_metric.label, fontsize=12, y=0.16)
    figure.supylabel(target.label, fontsize=12, x=0.016)
    present = [method for method in METHOD_ORDER if method in set(shown.method)]
    handles = [Line2D([0], [0], color=METHOD_STYLE[method][0], linewidth=1.1, marker=METHOD_STYLE[method][1], markersize=6, markerfacecolor=METHOD_STYLE[method][0] if METHOD_STYLE[method][2] else "white", markeredgewidth=1.4, label=method) for method in present]
    if target.baseline is not None:
        handles.append(Line2D([0], [0], color="#7B8794", linewidth=1.2, linestyle=(0, (5, 3)), label="SASRec"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False, handlelength=1.8, columnspacing=1.15)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.38, wspace=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{target.key}_vs_{sid_metric.stem}"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300 if extension == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return True


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for metric in SID_METRICS:
            for method in METHOD_ORDER:
                points = frame[frame.method.eq(method)].dropna(subset=[metric.column, target.column])
                rows.append({"target": target.key, "sid_metric": metric.column, "method": method, "num_points": len(points)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path)
    parser.add_argument("--item-logq-report", type=Path)
    parser.add_argument("--pure-sid-logq-report", type=Path)
    parser.add_argument("--poster-extensions-report", type=Path)
    parser.add_argument(
        "--poster-extensions-33148-report",
        type=Path,
        help="Completed 33,148-item PLUM/Balance export to add to the legacy report.",
    )
    parser.add_argument("--ranking-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/all_methods_sid_quality"))
    args = parser.parse_args()
    args.raw_report = args.raw_report or first_existing(
        Path("reports/figures/generation/raw_sid_recall/matched_runs.csv"),
        Path("reports/figures/generation/poster_raw_sid/matched_runs.csv"),
    )
    args.item_logq_report = args.item_logq_report or first_existing(
        Path("reports/figures/generation/rqvae_item_logq_recall/matched_logq_rqvae_runs.csv"),
        Path("reports/figures/generation/logq_rqvae_sid_quality_recall/matched_logq_rqvae_runs.csv"),
    )
    args.pure_sid_logq_report = args.pure_sid_logq_report or first_existing(
        Path("reports/figures/generation/rqvae_pure_sid_logq_recall/matched_logq_rqvae_runs.csv"),
    )
    args.poster_extensions_report = args.poster_extensions_report or first_existing(
        Path("reports/figures/generation/poster_plum_balance_recall/matched_generation_runs.csv"),
        Path("reports/figures/generation/poster_raw_sid_with_plum_balance/matched_generation_runs.csv"),
    )
    args.poster_extensions_33148_report = args.poster_extensions_33148_report or first_existing(
        Path("reports/figures/generation/poster_plum_balance_recall_33148/matched_generation_runs.csv"),
    )
    args.ranking_report = args.ranking_report or first_existing(
        Path("reports/figures/ranking/matched_runs.csv"),
        Path("reports/figures/ranking/sid_quality_correlations/matched_runs.csv"),
    )
    raw = normalize(read_csv(args.raw_report, "raw_sid"))
    item_logq = normalize(read_csv(args.item_logq_report, "item_logq", required=False), method="RQVAE + item LogQ")
    pure_sid_logq = normalize(read_csv(args.pure_sid_logq_report, "pure_sid_logq", required=False), method="RQVAE + pure SID LogQ")
    extensions = normalize(read_csv(args.poster_extensions_report, "poster_extensions", required=False))
    extensions_33148 = normalize(read_csv(args.poster_extensions_33148_report, "poster_extensions_33148", required=False))
    generation = deduplicate(numeric(filter_extensions(pd.concat([
        raw, item_logq, pure_sid_logq, extensions, extensions_33148,
    ], ignore_index=True))))
    ranking = deduplicate(numeric(filter_extensions(normalize(read_csv(args.ranking_report, "ranking")))))
    combined = pd.concat([generation, ranking], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generation.to_csv(args.output_dir / "generation_points.csv", index=False)
    ranking.to_csv(args.output_dir / "ranking_points.csv", index=False)
    coverage(combined).to_csv(args.output_dir / "metric_coverage.csv", index=False)
    written = sum(plot_metric(generation if target.key == "recall100" else ranking, metric, target, args.output_dir / target.key) for target in TARGETS for metric in SID_METRICS)
    print(f"Figures: {written} (expected: {len(TARGETS) * len(SID_METRICS)})")
    print(f"Output: {args.output_dir}")
    print("Recall methods:", ", ".join(method for method in METHOD_ORDER if method in set(generation.method)))
    print("Ranking methods:", ", ".join(method for method in METHOD_ORDER if method in set(ranking.method)))


if __name__ == "__main__":
    main()
