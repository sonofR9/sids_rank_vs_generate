#!/usr/bin/env python3
"""Build pooled Pearson and Spearman SID-quality correlation heatmaps.

The input may be the all-method report directory or its ZIP archive. Correlation
pairs are pooled across every available SID method and configuration, while the
generation and ranking downstream metrics remain separate heatmap rows.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str


SID_METRICS = (
    MetricSpec("icr", "ICR"),
    MetricSpec("collision_rate", "Collision\nRate"),
    MetricSpec("p95_collision_group_size", "p95 Collision\nGroup Size"),
    MetricSpec("g_max", "G_max"),
    MetricSpec("cur_total", "CUR_Total"),
    MetricSpec("code_gini", "Code Gini"),
    MetricSpec("code_entropy", "Code\nEntropy"),
    MetricSpec("intra_code_similarity", "Intra-code\nSimilarity"),
    MetricSpec("prefix_coherence_1", "Prefix\nCoherence@1"),
    MetricSpec("prefix_coherence_2", "Prefix\nCoherence@2"),
    MetricSpec("prefix_coherence_3", "Prefix\nCoherence@3*"),
    MetricSpec("valid_sid_rate", "Valid SID\nRate"),
)


@dataclass(frozen=True)
class TargetSpec:
    key: str
    column: str
    label: str
    task: str
    source_table: str


TARGETS = (
    TargetSpec("recall100", "recall_at_100", "Recall@100", "Generative", "generation_points.csv"),
    TargetSpec(
        "dcnv2",
        "dcnv2_pairwise_accuracy",
        "DCNv2 Pairwise Accuracy",
        "Ranking",
        "ranking_points.csv",
    ),
    TargetSpec(
        "dcnv2_roc_auc",
        "dcnv2_roc_auc",
        "DCNv2 ROC-AUC",
        "Ranking",
        "ranking_points.csv",
    ),
    TargetSpec(
        "transact",
        "transact_pairwise_accuracy",
        "TransAct Pairwise Accuracy",
        "Ranking",
        "ranking_points.csv",
    ),
    TargetSpec(
        "transact_roc_auc",
        "transact_roc_auc",
        "TransAct ROC-AUC",
        "Ranking",
        "ranking_points.csv",
    ),
)


def open_report_csv(input_path: Path, filename: str) -> BinaryIO:
    if input_path.is_dir():
        path = input_path / filename
        if not path.is_file():
            raise SystemExit(f"Input table not found: {path}")
        return path.open("rb")

    if not input_path.is_file():
        raise SystemExit(f"Input archive not found: {input_path}")
    archive = zipfile.ZipFile(input_path)
    matches = [name for name in archive.namelist() if Path(name).name == filename]
    if len(matches) != 1:
        archive.close()
        raise SystemExit(
            f"Expected exactly one {filename} in {input_path}, found {len(matches)}"
        )
    stream = archive.open(matches[0])
    original_close = stream.close

    def close() -> None:
        original_close()
        archive.close()

    stream.close = close  # type: ignore[method-assign]
    return stream


def resolve_input_path(input_path: Path) -> Path:
    """Fall back to an already exported report when a shared archive vanishes."""
    if input_path.exists():
        return input_path
    fallback = Path("reports/figures/all_methods_sid_quality")
    if all((fallback / name).is_file() for name in ("generation_points.csv", "ranking_points.csv")):
        print(f"Input unavailable; using existing report tables: {fallback}")
        return fallback
    return input_path


def read_points(input_path: Path, filename: str) -> pd.DataFrame:
    with open_report_csv(input_path, filename) as stream:
        frame = pd.read_csv(stream)
    numeric_columns = [
        target.column for target in TARGETS if target.source_table == filename
    ] + [metric.column for metric in SID_METRICS]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan)


def align_generation_columns(frame: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    for column in template.columns:
        if column not in aligned:
            aligned[column] = np.nan
    return aligned.reindex(columns=template.columns)


def augment_generation_from_local_reports(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover supplementary methods when the user-provided ZIP is unavailable."""
    additions: list[pd.DataFrame] = []
    present = set(frame["method"].dropna().astype(str))

    pure_sid_archive = Path("pure_sid_logq_recall_l2_l3_c256.tar.gz")
    pure_sid_method = "RQVAE + pure SID LogQ"
    if pure_sid_method not in present and pure_sid_archive.is_file():
        member_name = "sid_logq_sid_quality_recall/matched_logq_rqvae_runs.csv"
        with tarfile.open(pure_sid_archive, "r:gz") as archive:
            member = archive.extractfile(member_name)
            if member is not None:
                pure_sid = pd.read_csv(member)
                pure_sid["method"] = pure_sid_method
                pure_sid["source"] = "pure_sid_logq"
                additions.append(align_generation_columns(pure_sid, frame))

    extensions_path = Path(
        "reports/figures/generation/poster_raw_sid_with_plum_balance/"
        "matched_generation_runs.csv"
    )
    extension_methods = {"PLUM", "BalanceRQKMeans"}
    if not extension_methods.issubset(present) and extensions_path.is_file():
        extensions = pd.read_csv(extensions_path)
        extensions = extensions[extensions["method"].isin(extension_methods - present)].copy()
        extensions["source"] = "poster_extensions_33148"
        additions.append(align_generation_columns(extensions, frame))

    if not additions:
        return frame
    combined = pd.concat([frame, *additions], ignore_index=True)
    dedupe_columns = ["method", "num_levels", "codebook_size", "run_name"]
    return combined.drop_duplicates(dedupe_columns, keep="last").reset_index(drop=True)


def augment_ranking_roc_auc(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ROC-AUC columns omitted by the combined all-method point export."""
    roc_columns = ["dcnv2_roc_auc", "transact_roc_auc"]
    if all(column in frame and frame[column].notna().all() for column in roc_columns):
        return frame

    source_path = Path("reports/figures/ranking/sid_quality_correlations/matched_runs.csv")
    if not source_path.is_file():
        missing = ", ".join(column for column in roc_columns if column not in frame)
        raise SystemExit(f"ROC-AUC columns unavailable ({missing}); source not found: {source_path}")

    keys = ["method", "num_levels", "codebook_size"]
    source = pd.read_csv(source_path, usecols=[*keys, *roc_columns])
    if source.duplicated(keys).any():
        duplicates = source.loc[source.duplicated(keys, keep=False), keys]
        raise SystemExit(f"Duplicate ranking ROC-AUC keys:\n{duplicates.to_string(index=False)}")

    output = frame.drop(columns=[column for column in roc_columns if column in frame])
    output = output.merge(source, on=keys, how="left", validate="one_to_one")
    missing_rows = output[output[roc_columns].isna().any(axis=1)]
    if not missing_rows.empty:
        raise SystemExit(
            "Missing ROC-AUC values for ranking configurations:\n"
            + missing_rows[keys].to_string(index=False)
        )
    return output


def filter_by_generation_recall(
    tables: dict[str, pd.DataFrame], maximum_recall: float
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Keep configurations at or below a generation Recall threshold."""
    keys = ["method", "num_levels", "codebook_size"]
    generation = tables["generation_points.csv"].copy()
    recall_by_key = generation[keys + ["recall_at_100"]].drop_duplicates(keys)
    if recall_by_key.duplicated(keys).any():
        raise SystemExit("Generation configurations are not unique for baseline filtering")

    generation_keep = generation["recall_at_100"].le(maximum_recall)
    filtered_generation = generation[generation_keep].copy()

    ranking = tables["ranking_points.csv"].drop(columns=["recall_at_100"], errors="ignore")
    ranking = ranking.merge(recall_by_key, on=keys, how="left", validate="one_to_one")
    missing = ranking[ranking["recall_at_100"].isna()]
    if not missing.empty:
        raise SystemExit(
            "Ranking configurations missing generation Recall for baseline filtering:\n"
            + missing[keys].to_string(index=False)
        )
    ranking_keep = ranking["recall_at_100"].le(maximum_recall)
    filtered_ranking = ranking[ranking_keep].copy()

    counts = {
        "generation_before": len(generation),
        "generation_after": len(filtered_generation),
        "ranking_before": len(tables["ranking_points.csv"]),
        "ranking_after": len(filtered_ranking),
    }
    return {
        "generation_points.csv": filtered_generation,
        "ranking_points.csv": filtered_ranking,
    }, counts


def correlation_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        frame = tables[target.source_table]
        for metric in SID_METRICS:
            pairs = frame[[metric.column, target.column]].dropna()
            pearson = spearman = kendall = np.nan
            reason = ""
            if len(pairs) < 2:
                reason = "fewer than 2 complete pairs"
            elif pairs[metric.column].nunique() < 2:
                reason = "SID metric is constant"
            elif pairs[target.column].nunique() < 2:
                reason = "downstream metric is constant"
            else:
                pearson = float(pearsonr(pairs[metric.column], pairs[target.column]).statistic)
                spearman = float(spearmanr(pairs[metric.column], pairs[target.column]).statistic)
                kendall = float(kendalltau(pairs[metric.column], pairs[target.column]).statistic)
            rows.append(
                {
                    "task": target.task,
                    "target": target.key,
                    "target_label": target.label,
                    "sid_metric": metric.column,
                    "sid_metric_label": metric.label.replace("\n", " "),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "kendall_tau": kendall,
                    "num_points": len(pairs),
                    "undefined_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def matrix(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    values = frame.pivot(index="target", columns="sid_metric", values=value_column)
    return values.reindex(
        index=[target.key for target in TARGETS],
        columns=[metric.column for metric in SID_METRICS],
    )


def annotations(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    lookup = frame.set_index(["target", "sid_metric"])
    labels = np.empty((len(TARGETS), len(SID_METRICS)), dtype=object)
    for row, target in enumerate(TARGETS):
        for column, metric in enumerate(SID_METRICS):
            item = lookup.loc[(target.key, metric.column)]
            value = item[value_column]
            labels[row, column] = (
                f"{value:+.2f}\nn={int(item['num_points'])}"
                if pd.notna(value)
                else f"NA\nn={int(item['num_points'])}"
            )
    return labels


def correlation_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sid_correlation",
        ("#2B6CB0", "#DCEAF5", "#F7F7F4", "#F7D6D0", "#C73E32"),
        N=256,
    )


def correlation_norm(values: pd.DataFrame) -> Normalize:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.01)
        return Normalize(vmin=minimum - padding, vmax=maximum + padding)
    if minimum < 0.0 < maximum:
        return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return Normalize(vmin=minimum, vmax=maximum)


def colorbar_ticks(image: object) -> list[float]:
    norm = image.norm
    ticks = [float(norm.vmin)]
    if float(norm.vmin) < 0.0 < float(norm.vmax):
        ticks.append(0.0)
    ticks.append(float(norm.vmax))
    return ticks


def draw_heatmap(
    axis: plt.Axes,
    values: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    *,
    show_y_labels: bool,
) -> None:
    masked = np.ma.masked_invalid(values.to_numpy(dtype=float))
    cmap = correlation_cmap()
    cmap.set_bad("#E5E7EB")
    image = axis.imshow(masked, cmap=cmap, norm=correlation_norm(values), aspect="auto")

    axis.set_xticks(np.arange(len(SID_METRICS)), [metric.label for metric in SID_METRICS])
    axis.set_yticks(
        np.arange(len(TARGETS)),
        [f"{target.task} · {target.label}" for target in TARGETS] if show_y_labels else [],
    )
    axis.tick_params(axis="x", labelrotation=35, length=0, pad=8)
    axis.tick_params(axis="y", length=0, pad=8)
    plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
    axis.set_title(title, fontsize=15, fontweight="bold", pad=18)
    axis.set_xlabel("SID quality metrics", fontsize=11, labelpad=12)

    axis.set_xticks(np.arange(-0.5, len(SID_METRICS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(TARGETS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row in range(len(TARGETS)):
        for column in range(len(SID_METRICS)):
            value = values.iat[row, column]
            normalized = image.norm(value) if pd.notna(value) else 0.5
            color = (
                "#FFFFFF"
                if pd.notna(value) and (normalized <= 0.17 or normalized >= 0.83)
                else "#17202A"
            )
            if pd.isna(value):
                color = "#5F6B76"
            axis.text(
                column,
                row,
                labels[row, column],
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
                linespacing=1.25,
            )
    return image


def save_single_heatmap(
    correlations: pd.DataFrame,
    value_column: str,
    title: str,
    output_stem: Path,
    sample_note: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_height = max(6.8, 1.15 * len(TARGETS) + 2.0)
    figure, axis = plt.subplots(figsize=(15.2, figure_height), facecolor="white")
    values = matrix(correlations, value_column)
    image = draw_heatmap(axis, values, annotations(correlations, value_column), title, show_y_labels=True)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
    colorbar.set_label("Correlation coefficient", rotation=270, labelpad=17)
    colorbar.set_ticks(colorbar_ticks(image))
    figure.text(
        0.5,
        0.01,
        f"* Prefix Coherence@3 · {sample_note} · colors span the observed min-max · cell text: coefficient and complete-pair count",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5F6B76",
    )
    figure.subplots_adjust(left=0.22, right=0.94, top=0.88, bottom=0.25)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def save_combined_heatmap(
    correlations: pd.DataFrame, output_stem: Path, sample_note: str
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(16.0, 19.0), facecolor="white")
    specifications = (
        ("spearman_rho", "Spearman correlation"),
        ("pearson_r", "Pearson correlation"),
        ("kendall_tau", "Kendall correlation"),
    )
    images = []
    for axis, (value_column, title) in zip(axes, specifications, strict=True):
        images.append(draw_heatmap(
            axis,
            matrix(correlations, value_column),
            annotations(correlations, value_column),
            title,
            show_y_labels=True,
        ))
    figure.subplots_adjust(left=0.20, right=0.88, top=0.97, bottom=0.08, hspace=0.52)
    colorbar_bounds = (
        (0.91, 0.70, 0.015, 0.18),
        (0.91, 0.41, 0.015, 0.18),
        (0.91, 0.12, 0.015, 0.18),
    )
    for image, bounds in zip(images, colorbar_bounds, strict=True):
        colorbar_axis = figure.add_axes(bounds)
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Correlation coefficient", rotation=270, labelpad=17)
        colorbar.set_ticks(colorbar_ticks(image))
    figure.text(
        0.5,
        0.01,
        f"* Prefix Coherence@3 · {sample_note} · colors span each panel's observed min-max · cell text: coefficient and complete-pair count",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5F6B76",
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def write_readme(
    output_dir: Path,
    input_path: Path,
    correlations: pd.DataFrame,
    maximum_recall: float | None,
) -> None:
    undefined = correlations[
        correlations.pearson_r.isna()
        | correlations.spearman_rho.isna()
        | correlations.kendall_tau.isna()
    ]
    readme = f"""# All-method SID quality correlation heatmaps

The heatmaps pool every available SID method and configuration from
`{input_path.name}`. Columns are SID quality metrics. Rows separate the
generative target (`Recall@100`) from four ranking targets: pairwise accuracy
and ROC-AUC for both `DCNv2` and `TransAct`.

Each cell shows the correlation coefficient and the number of complete pairs
used for that coefficient. Pearson, Spearman, and Kendall tau are calculated independently
after dropping rows missing either metric. Gray `NA` cells indicate a constant
input or fewer than two complete pairs. Each heatmap uses its observed minimum
and maximum as the color limits, with zero kept at the neutral midpoint.
`Prefix Coherence@3` is marked with an asterisk on the X axis.
{f'Only configurations with generation Recall@100 <= {maximum_recall:g} are included. The same configuration keys filter the ranking rows.' if maximum_recall is not None else ''}

Outputs:

- `spearman_heatmap.*`: Spearman rank correlation.
- `pearson_heatmap.*`: Pearson linear correlation.
- `kendall_heatmap.*`: Kendall rank correlation.
- `correlation_heatmaps.*`: all three heatmaps in one figure.
- `correlations.csv`: long-form coefficients, pair counts, and undefined reasons.
- `pearson_matrix.csv`, `spearman_matrix.csv`, `kendall_matrix.csv`, `num_points_matrix.csv`: compact matrices.

Undefined cells: {len(undefined)} of {len(correlations)}.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="All-method report directory or ZIP archive")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures/all_methods_sid_quality_heatmaps"),
    )
    parser.add_argument(
        "--max-generation-recall",
        type=float,
        help="Keep only configurations whose generation Recall@100 is at or below this value.",
    )
    args = parser.parse_args()

    args.input = resolve_input_path(args.input)

    tables = {
        filename: read_points(args.input, filename)
        for filename in {target.source_table for target in TARGETS}
    }
    tables["generation_points.csv"] = augment_generation_from_local_reports(
        tables["generation_points.csv"]
    )
    tables["ranking_points.csv"] = augment_ranking_roc_auc(tables["ranking_points.csv"])
    filter_counts = None
    if args.max_generation_recall is not None:
        tables, filter_counts = filter_by_generation_recall(
            tables, args.max_generation_recall
        )
    correlations = correlation_rows(tables)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_note = (
        f"Recall@100 <= {args.max_generation_recall:g}"
        if args.max_generation_recall is not None
        else "all available configurations"
    )

    correlations.to_csv(args.output_dir / "correlations.csv", index=False)
    matrix(correlations, "pearson_r").to_csv(args.output_dir / "pearson_matrix.csv")
    matrix(correlations, "spearman_rho").to_csv(args.output_dir / "spearman_matrix.csv")
    matrix(correlations, "kendall_tau").to_csv(args.output_dir / "kendall_matrix.csv")
    matrix(correlations, "num_points").to_csv(args.output_dir / "num_points_matrix.csv")
    for filename, frame in tables.items():
        frame.to_csv(args.output_dir / filename, index=False)

    save_single_heatmap(
        correlations,
        "spearman_rho",
        "Spearman correlation: downstream quality vs SID quality",
        args.output_dir / "spearman_heatmap",
        sample_note,
    )
    save_single_heatmap(
        correlations,
        "pearson_r",
        "Pearson correlation: downstream quality vs SID quality",
        args.output_dir / "pearson_heatmap",
        sample_note,
    )
    save_single_heatmap(
        correlations,
        "kendall_tau",
        "Kendall correlation: downstream quality vs SID quality",
        args.output_dir / "kendall_heatmap",
        sample_note,
    )
    save_combined_heatmap(
        correlations, args.output_dir / "correlation_heatmaps", sample_note
    )
    write_readme(
        args.output_dir, args.input, correlations, args.max_generation_recall
    )

    metadata = {
        "input": str(args.input.resolve()),
        "pooling": "all available SID methods and configurations",
        "targets": [target.key for target in TARGETS],
        "sid_metrics": [metric.column for metric in SID_METRICS],
        "generation_rows": len(tables["generation_points.csv"]),
        "ranking_rows": len(tables["ranking_points.csv"]),
        "maximum_generation_recall": args.max_generation_recall,
        "filter_counts": filter_counts,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Output: {args.output_dir}")
    print(f"Cells: {len(correlations)}")
    print(f"Undefined Pearson: {correlations.pearson_r.isna().sum()}")
    print(f"Undefined Spearman: {correlations.spearman_rho.isna().sum()}")
    print(f"Undefined Kendall: {correlations.kendall_tau.isna().sum()}")


if __name__ == "__main__":
    main()
