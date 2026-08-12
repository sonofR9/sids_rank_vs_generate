#!/usr/bin/env python3
"""Plot Pearson and Spearman correlations for every available SID metric."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = Path(
    "reports/figures/"
    "all_methods_sid_quality_fixed_transact_extended_sid_metrics"
)
DEFAULT_DCNV2_SOURCE = Path("reports/figures/ranking/matched_runs.csv")
KEY_COLUMNS = ["method", "num_levels", "codebook_size"]
ORIGINAL_SID_METRICS = (
    "icr",
    "collision_rate",
    "p95_collision_group_size",
    "g_max",
    "cur_total",
    "code_gini",
    "code_entropy",
    "intra_code_similarity",
    "prefix_coherence_1",
    "prefix_coherence_2",
    "prefix_coherence_3",
    "valid_sid_rate",
)


@dataclass(frozen=True)
class Target:
    key: str
    column: str
    label: str
    source: str


TARGETS = (
    Target("recall100", "recall_at_100", "Generation · Recall@100", "generation"),
    Target(
        "dcnv2_pairwise",
        "dcnv2_pairwise_accuracy",
        "DCNv2 · Pairwise accuracy",
        "ranking",
    ),
    Target("dcnv2_roc_auc", "dcnv2_roc_auc", "DCNv2 · ROC-AUC", "ranking"),
    Target(
        "transact_pairwise",
        "transact_pairwise_accuracy",
        "TransAct · Pairwise accuracy",
        "ranking",
    ),
    Target(
        "transact_roc_auc",
        "transact_roc_auc",
        "TransAct · ROC-AUC",
        "ranking",
    ),
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def metric_label(column: str) -> str:
    fixed = {
        "icr": "ICR",
        "p95_collision_group_size": "p95 collision group size",
        "g_max": "G_max",
        "cur_total": "CUR_Total",
        "intra_code_similarity": "Intra-code similarity",
        "intra_code_similarity_v2": "Intra-code similarity v2",
    }
    if column in fixed:
        return fixed[column]
    label = column.replace("_", " ")
    label = label.replace("adjusted semantic variance explained", "Adjusted SVE")
    label = label.replace("normalized conditional entropy", "Normalized CE")
    label = label.replace("mean active branch fraction", "Mean active branch")
    label = label.replace("codebook utilization rate", "Codebook utilization")
    label = label.replace("pair refinement rate", "Pair refinement")
    label = label.replace("prefix coherence ", "Prefix coherence@")
    for level in range(1, 5):
        if label.endswith(f" {level}"):
            label = f"{label[:-2]}@{level}"
            break
    return label.capitalize()


def load_metric_columns(report: Path) -> list[str]:
    metadata_path = report / "extended_sid_metrics_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    extended = list(metadata["extended_metrics"])
    metrics = [*ORIGINAL_SID_METRICS, *extended]
    if len(metrics) != len(set(metrics)):
        raise ValueError("duplicate SID metric names in report metadata")
    return metrics


def add_dcnv2_roc_auc(ranking: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    if "dcnv2_roc_auc" in ranking:
        return ranking
    source = pd.read_csv(source_path, usecols=[*KEY_COLUMNS, "dcnv2_roc_auc"])
    if source.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"duplicate DCNv2 keys in {source_path}")
    result = ranking.merge(source, on=KEY_COLUMNS, how="left", validate="one_to_one")
    # Use the same completed configurations for both DCNv2 ranking targets.
    result.loc[result["dcnv2_pairwise_accuracy"].isna(), "dcnv2_roc_auc"] = np.nan
    return result


def calculate_correlations(
    generation: pd.DataFrame,
    ranking: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    tables = {"generation": generation, "ranking": ranking}
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        frame = tables[target.source]
        for metric in metrics:
            pairs = frame[[metric, target.column]].apply(
                pd.to_numeric, errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()
            reason = ""
            pearson = spearman = np.nan
            if len(pairs) < 2:
                reason = "fewer than 2 complete pairs"
            elif pairs[metric].nunique() < 2:
                reason = "SID metric is constant"
            elif pairs[target.column].nunique() < 2:
                reason = "downstream metric is constant"
            else:
                pearson = float(pairs[metric].corr(pairs[target.column], method="pearson"))
                spearman = float(pairs[metric].corr(pairs[target.column], method="spearman"))
            rows.append(
                {
                    "target": target.key,
                    "target_label": target.label,
                    "sid_metric": metric,
                    "sid_metric_label": metric_label(metric),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "num_points": len(pairs),
                    "undefined_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def correlation_matrix(
    correlations: pd.DataFrame,
    metrics: list[str],
    value_column: str,
) -> pd.DataFrame:
    return correlations.pivot(
        index="target", columns="sid_metric", values=value_column
    ).reindex(
        index=[target.key for target in TARGETS],
        columns=metrics,
    )


def correlation_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sid_correlation",
        ("#265D97", "#D5E5F2", "#F7F7F4", "#F4D2CA", "#B9362B"),
        N=256,
    )


def plot_heatmap(
    correlations: pd.DataFrame,
    metrics: list[str],
    value_column: str,
    title: str,
    output_stem: Path,
) -> None:
    values = correlation_matrix(correlations, metrics, value_column)
    counts = correlation_matrix(correlations, metrics, "num_points")
    figure_width = max(28.0, 0.54 * len(metrics))
    figure, axis = plt.subplots(figsize=(figure_width, 8.0), facecolor="white")
    cmap = correlation_cmap()
    cmap.set_bad("#E5E7EB")
    image = axis.imshow(
        np.ma.masked_invalid(values.to_numpy(dtype=float)),
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(metrics)),
        [metric_label(metric) for metric in metrics],
        rotation=58,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_yticks(
        np.arange(len(TARGETS)),
        [target.label for target in TARGETS],
    )
    axis.tick_params(axis="both", length=0, pad=7, labelsize=8.2)
    axis.set_title(title, fontsize=17, fontweight="bold", pad=18)
    axis.set_xlabel("SID metrics", fontsize=12, labelpad=15)
    axis.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(TARGETS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.3)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row in range(len(TARGETS)):
        for column in range(len(metrics)):
            value = values.iat[row, column]
            count = int(counts.iat[row, column])
            label = f"{value:+.2f}\nn={count}" if pd.notna(value) else f"NA\nn={count}"
            text_color = "white" if pd.notna(value) and abs(value) >= 0.68 else "#17202A"
            if pd.isna(value):
                text_color = "#5F6B76"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=5.8,
                color=text_color,
                linespacing=1.1,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.012, pad=0.012)
    colorbar.set_label("Correlation coefficient", rotation=270, labelpad=17)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    figure.text(
        0.5,
        0.01,
        "Cells show coefficient and complete-pair count; gray cells are undefined.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5F6B76",
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.36)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dcnv2-source", type=Path, default=DEFAULT_DCNV2_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    report = resolve(args.report_dir)
    dcnv2_source = resolve(args.dcnv2_source)
    output = resolve(args.output_dir) if args.output_dir else report / "correlation_heatmaps"
    metrics = load_metric_columns(report)
    generation = pd.read_csv(report / "generation_points.csv")
    ranking = add_dcnv2_roc_auc(
        pd.read_csv(report / "ranking_points.csv"), dcnv2_source
    )
    missing = [
        column
        for column in metrics
        if column not in generation or column not in ranking
    ]
    if missing:
        raise ValueError(f"SID metric columns missing from report: {missing}")

    correlations = calculate_correlations(generation, ranking, metrics)
    output.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(output / "correlations.csv", index=False)
    for value_column, stem, title in (
        ("pearson_r", "pearson_heatmap", "Pearson correlation: downstream quality vs SID metrics"),
        ("spearman_rho", "spearman_heatmap", "Spearman correlation: downstream quality vs SID metrics"),
    ):
        correlation_matrix(correlations, metrics, value_column).to_csv(
            output / f"{stem.replace('_heatmap', '')}_matrix.csv"
        )
        plot_heatmap(correlations, metrics, value_column, title, output / stem)
    correlation_matrix(correlations, metrics, "num_points").to_csv(
        output / "num_points_matrix.csv"
    )

    metadata = {
        "report": str(report.relative_to(REPOSITORY)),
        "dcnv2_roc_auc_source": str(dcnv2_source.relative_to(REPOSITORY)),
        "targets": [target.key for target in TARGETS],
        "sid_metrics": metrics,
        "sid_metric_count": len(metrics),
        "cells": len(correlations),
        "generation_rows": len(generation),
        "ranking_rows": len(ranking),
        "target_complete_rows": {
            target.key: int(
                (generation if target.source == "generation" else ranking)[
                    target.column
                ].notna().sum()
            )
            for target in TARGETS
        },
        "undefined_pearson": int(correlations["pearson_r"].isna().sum()),
        "undefined_spearman": int(correlations["spearman_rho"].isna().sum()),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Extended SID correlation heatmaps\n\n"
        "Pearson and Spearman correlations are pooled across all available SID "
        "methods and configurations. The five rows are generation Recall@100, "
        "then pairwise accuracy and ROC-AUC for DCNv2 and TransAct. The 56 "
        "columns contain all 12 original and 44 newly calculated SID metrics.\n\n"
        "Each cell reports the correlation coefficient and complete-pair count. "
        "DCNv2 uses the 28 configurations available in the combined report; "
        "generation and TransAct use all 44. Gray cells are undefined because "
        "the SID metric is constant or has fewer than two complete pairs.\n\n"
        "Outputs are written as PNG, PDF, and SVG, with long-form correlations "
        "and matrix CSV files alongside them.\n",
        encoding="utf-8",
    )
    print(
        f"COMPLETE metrics={len(metrics)} targets={len(TARGETS)} "
        f"cells={len(correlations)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
