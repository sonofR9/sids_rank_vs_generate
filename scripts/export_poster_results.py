#!/usr/bin/env python3
"""Export poster runs from MLflow and draw Recall@100 versus raw ICR."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


METHOD_COLORS = {
    "RQKMeans": "#4C78A8",
    "RQVAE": "#F58518",
    "RQVAE-token": "#F58518",
    "RQVAE (RRS)": "#E45756",
    "RQVAE (ECM)": "#B279A2",
    "R3VAE (spherical)": "#54A24B",
    "R3VAE (rqkmeans)": "#72B7B2",
}

METHOD_MARKER_STYLES = {
    "RQVAE (RRS)": {
        "marker": "D",
        "filled": False,
        "markersize": 9.0,
        "edgewidth": 1.9,
        "zorder": 5,
    },
    "RQVAE (ECM)": {
        "marker": "o",
        "filled": False,
        "markersize": 7.8,
        "edgewidth": 1.8,
        "zorder": 6,
    },
    "RQVAE-token": {
        "marker": "s",
        "filled": True,
        "markersize": 6.2,
        "edgewidth": 1.5,
        "zorder": 7,
    },
}

DEFAULT_MARKER_STYLE = {
    "marker": "o",
    "filled": False,
    "markersize": 7.5,
    "edgewidth": 1.8,
    "zorder": 3,
}

METHOD_ORDER = [
    "R3VAE (rqkmeans)",
    "R3VAE (spherical)",
    "RQKMeans",
    "RQVAE (ECM)",
    "RQVAE",
    "RQVAE (RRS)",
    "RQVAE-token",
]

# These methods were explicitly measured as discrete L/C sweeps. Token-based
# collision resolution is a separate RQVAE configuration and stays isolated.
CONNECTED_SWEEP_METHODS = {
    "R3VAE (rqkmeans)",
    "R3VAE (spherical)",
    "RQKMeans",
    "RQVAE (ECM)",
    "RQVAE",
    "RQVAE (RRS)",
}

def _value(row: pd.Series, name: str, default=None):
    value = row.get(name, default)
    return default if pd.isna(value) else value


def _child_by_parent(runs: pd.DataFrame) -> dict[str, pd.Series]:
    result = {}
    parent_column = "tags.mlflow.parentRunId"
    if parent_column not in runs:
        return result
    for _, row in runs.iterrows():
        parent = _value(row, parent_column)
        if parent and _value(row, "tags.stage") == "sid":
            result[str(parent)] = row
    return result


def _model_name(parent: pd.Series, child: pd.Series | None) -> str:
    tagged = _value(parent, "tags.sid_model")
    if tagged:
        return str(tagged).lower()
    pipeline = _value(parent, "tags.pipeline")
    if pipeline is None and child is not None:
        pipeline = _value(child, "tags.pipeline")
    return {
        "RQKMeansPipeline": "rqkmeans",
        "RQVAEPipeline": "rqvae",
        "R3VAESid": "r3vae",
    }.get(str(pipeline), "unknown")


def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_int(value) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _sid_parameter(parent: pd.Series, child: pd.Series | None, name: str, default=None):
    tagged = _value(parent, f"tags.{name}")
    if tagged is not None:
        return tagged
    if child is not None:
        return _value(child, f"params.config.{name}", default)
    return default


def _codebook_shape(parent: pd.Series, child: pd.Series | None) -> tuple[int, int]:
    size = _sid_parameter(parent, child, "codebook_size")
    levels = _sid_parameter(parent, child, "num_levels")
    sizes = _sid_parameter(parent, child, "codebook_sizes")
    if sizes is not None:
        parsed = json.loads(sizes) if isinstance(sizes, str) else list(sizes)
        if parsed:
            size = parsed[0]
            levels = len(parsed)
    return int(size or 0), int(levels or 0)


def _records(runs: pd.DataFrame) -> pd.DataFrame:
    children = _child_by_parent(runs)
    records = []
    for _, row in runs.iterrows():
        run_id = str(row["run_id"])
        child = children.get(run_id)
        base_icr = _value(row, "metrics.sid.details.assignment.base_icr")
        final_icr = _value(row, "metrics.sid.independent_code_rate")
        icr = final_icr if base_icr is None else base_icr
        recall = _value(row, "metrics.evaluations.generation.recall_at_100")
        if icr is None or recall is None:
            continue
        model = _model_name(row, child)
        if model == "unknown":
            continue
        codebook_size, num_levels = _codebook_shape(row, child)
        init = _sid_parameter(row, child, "init")
        if init is None:
            init = _sid_parameter(row, child, "kmeans_init_method")
        if init is None:
            init = "rqkmeans" if model in {"rqkmeans", "rqvae"} else "unknown"
        assignment = _sid_parameter(row, child, "assignment")
        if assignment is None:
            assignment = _sid_parameter(
                row, child, "assignment_method", "nearest"
            )
        collision_resolution = _value(row, "tags.collision_resolution")
        if collision_resolution is None:
            resolved = _value(
                row,
                "params.training.generation.resolve_collisions",
                False,
            )
            collision_resolution = "token" if _bool(resolved) else "none"
        max_rows = _value(row, "tags.max_rows")
        if max_rows is None:
            max_rows = _value(row, "params.max_rows")
        records.append(
            {
                "model": model,
                "init": init,
                "assignment": assignment,
                "collision_token": collision_resolution == "token",
                "codebook_size": codebook_size,
                "num_levels": num_levels,
                "seed": int(_sid_parameter(row, child, "seed", 42)),
                "max_rows": _optional_int(max_rows),
                "icr": float(icr),
                "collision_rate": 1.0 - float(icr),
                "final_icr": float(final_icr),
                "recall_100": float(recall),
                "mlflow_run_id": run_id,
                "start_time": row.get("start_time"),
                "poster_tagged": _value(row, "tags.poster") == "true",
            }
        )
    return pd.DataFrame.from_records(records)


def _family(row: pd.Series) -> str:
    if row["model"] == "r3vae":
        return f"R3VAE ({row['init']})"
    if row["model"] == "rqvae" and row["assignment"] != "nearest":
        return f"RQVAE ({row['assignment'].upper()})"
    return {"rqkmeans": "RQKMeans", "rqvae": "RQVAE"}.get(
        row["model"], row["model"]
    )


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    configuration = [
        "model",
        "init",
        "assignment",
        "collision_token",
        "codebook_size",
        "num_levels",
    ]
    latest_per_seed = (
        frame.sort_values("start_time")
        .drop_duplicates(configuration + ["seed"], keep="last")
    )
    return (
        latest_per_seed.groupby(configuration, dropna=False, as_index=False)
        .agg(
            icr=("icr", "mean"),
            collision_rate=("collision_rate", "mean"),
            final_icr=("final_icr", "mean"),
            recall_100=("recall_100", "mean"),
            recall_100_std=("recall_100", "std"),
            seed_count=("seed", "nunique"),
        )
        .fillna({"recall_100_std": 0.0})
    )


def _load_sasrec_baseline(metrics_path: Path | None) -> float | None:
    if metrics_path is None:
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    max_seq_len = int(config.get("max_seq_len", 0))
    topk = int(config.get("topk", 0))
    if max_seq_len != 16:
        raise ValueError(
            f"SASRec baseline must use max_seq_len=16, received {max_seq_len}"
        )
    if topk != 100:
        raise ValueError(f"SASRec baseline must use topk=100, received {topk}")
    metrics = payload.get("metrics", {})
    recall = metrics.get("recall", metrics.get("recall@100"))
    if recall is None:
        raise ValueError("SASRec metrics JSON does not contain Recall@100")
    recall = float(recall)
    if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
        raise ValueError(f"Invalid SASRec Recall@100: {recall}")
    return recall


def _plot(
    frame: pd.DataFrame,
    output_dir: Path,
    sasrec_recall: float | None = None,
) -> None:
    plot_frame = frame.copy()
    plot_frame["family"] = plot_frame.apply(_family, axis=1)
    plot_frame["method"] = plot_frame.apply(
        lambda row: (
            f"{row['family']}-token"
            if bool(row["collision_token"])
            else row["family"]
        ),
        axis=1,
    )
    method_order = list(METHOD_ORDER)
    method_order.extend(
        sorted(set(plot_frame["method"]) - set(method_order))
    )

    level_frame = plot_frame[plot_frame["codebook_size"] == 256].copy()
    codebook_frame = plot_frame[plot_frame["num_levels"] == 3].copy()
    shown = pd.concat([level_frame, codebook_frame], ignore_index=True)
    if shown.empty:
        raise ValueError("No L- or C-sweep points are available for plotting")

    x_min = float(shown["icr"].min())
    x_max = float(shown["icr"].max())
    y_min = float(shown["recall_100"].min())
    y_max = float(shown["recall_100"].max())
    if sasrec_recall is not None:
        y_min = min(y_min, sasrec_recall)
        y_max = max(y_max, sasrec_recall)
    x_limits = (
        max(0.0, min(0.38, x_min - 0.025)),
        min(1.02, max(1.0, x_max + 0.02)),
    )
    y_limits = (
        max(0.0, min(0.062, y_min - 0.003)),
        min(1.0, max(0.130, y_max + 0.003)),
    )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    # Reserve distinct vertical bands for axes, the global X label, and the
    # two-row method legend. This is deterministic across Matplotlib versions.
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.275,
        wspace=0.05,
    )

    def draw_panel(
        axis: plt.Axes,
        panel_frame: pd.DataFrame,
        *,
        parameter: str,
        title: str,
        sweep_label: str,
        sweep_label_xy: tuple[float, float],
        show_baseline_label: bool,
    ) -> None:
        for method in method_order:
            points = panel_frame[panel_frame["method"] == method].sort_values(
                parameter
            )
            if points.empty:
                continue
            color = METHOD_COLORS.get(
                method,
                METHOD_COLORS.get(
                    str(points.iloc[0]["family"]),
                    "#666666",
                ),
            )
            marker_style = {
                **DEFAULT_MARKER_STYLE,
                **METHOD_MARKER_STYLES.get(method, {}),
            }

            # Token points are intentionally shown as references rather than
            # connected to the open-circle SID sweeps.
            if len(points) >= 2 and method in CONNECTED_SWEEP_METHODS:
                axis.plot(
                    points["icr"],
                    points["recall_100"],
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    alpha=0.92,
                    zorder=2,
                )

            for _, point in points.iterrows():
                axis.errorbar(
                    point["icr"],
                    point["recall_100"],
                    yerr=(
                        point["recall_100_std"]
                        if point["seed_count"] > 1
                        else None
                    ),
                    marker=marker_style["marker"],
                    markersize=marker_style["markersize"],
                    markerfacecolor=(
                        color if marker_style["filled"] else "white"
                    ),
                    markeredgecolor=color,
                    markeredgewidth=marker_style["edgewidth"],
                    color=color,
                    elinewidth=1.2,
                    capsize=3,
                    linestyle="none",
                    zorder=marker_style["zorder"],
                )

        if sasrec_recall is not None:
            baseline_color = "#6B7280"
            axis.axhline(
                sasrec_recall,
                color=baseline_color,
                linewidth=1.15,
                linestyle=(0, (4, 3)),
                alpha=0.82,
                zorder=1,
            )
            if show_baseline_label:
                axis.text(
                    x_limits[0] + 0.09 * (x_limits[1] - x_limits[0]),
                    sasrec_recall + 0.00055,
                    "SASRec baseline",
                    ha="left",
                    va="bottom",
                    fontsize=8.0,
                    color=baseline_color,
                    zorder=4,
                )

        axis.set_title(title, pad=9, fontweight="normal")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.xaxis.set_major_locator(MultipleLocator(0.1))
        axis.yaxis.set_major_locator(MultipleLocator(0.01))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.grid(
            True,
            which="major",
            color="#D1D5DB",
            linewidth=0.65,
            alpha=0.42,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9CA3AF")
        axis.spines["bottom"].set_color("#9CA3AF")
        axis.tick_params(colors="#374151", width=0.8)
        axis.text(
            0.992,
            0.985,
            "better ↗",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            color="#6B7280",
            alpha=0.72,
        )
        axis.text(
            *sweep_label_xy,
            sweep_label,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=8,
            color="#4B5563",
            alpha=0.86,
        )

    draw_panel(
        axes[0],
        level_frame,
        parameter="num_levels",
        title="Varying number of codebooks (C=256)",
        sweep_label="L: 2 → 3 → 4",
        sweep_label_xy=(0.36, 0.91),
        show_baseline_label=False,
    )
    draw_panel(
        axes[1],
        codebook_frame,
        parameter="codebook_size",
        title="Varying codebook size (L=3)",
        sweep_label="C: 128 → 256 → 512 → 1024",
        sweep_label_xy=(0.04, 0.91),
        show_baseline_label=True,
    )
    figure.supxlabel(
        "Raw independent code rate (ICR)",
        fontsize=11,
        y=0.175,
    )
    figure.supylabel("Recall@100", fontsize=11, x=0.018)

    present_methods = [
        method for method in method_order if method in set(shown["method"])
    ]
    legend_handles = []
    for method in present_methods:
        color = METHOD_COLORS.get(method, "#666666")
        marker_style = {
            **DEFAULT_MARKER_STYLE,
            **METHOD_MARKER_STYLES.get(method, {}),
        }
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="none" if method.endswith("-token") else color,
                linewidth=2.0,
                marker=marker_style["marker"],
                markersize=min(float(marker_style["markersize"]), 7.2),
                markerfacecolor=(
                    color if marker_style["filled"] else "white"
                ),
                markeredgecolor=color,
                markeredgewidth=marker_style["edgewidth"],
                label=method,
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.25,
    )

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"recall100_vs_icr.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", default="sid-comparison")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/poster"))
    parser.add_argument(
        "--sasrec-metrics-json",
        type=Path,
        help=(
            "SASRec metrics JSON with max_seq_len=16 and topk=100. "
            "Adds a horizontal Recall@100 baseline and an ICR=1 marker."
        ),
    )
    parser.add_argument(
        "--poster-only",
        action="store_true",
        help="Exclude compatible historical runs that do not have poster=true.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"MLflow experiment not found: {args.experiment_name}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )
    frame = _records(runs)
    if not frame.empty:
        frame = frame[frame["max_rows"].isna()]
    if args.poster_only and not frame.empty:
        frame = frame[frame["poster_tagged"]]
    if frame.empty:
        raise SystemExit("No completed poster runs with ICR and Recall@100 found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.sort_values(
        ["model", "init", "assignment", "collision_token", "icr"],
        inplace=True,
    )
    frame.to_csv(args.output_dir / "poster_results.csv", index=False)
    summary = _summarize(frame)
    summary.to_csv(args.output_dir / "poster_results_summary.csv", index=False)
    sasrec_recall = _load_sasrec_baseline(args.sasrec_metrics_json)
    _plot(summary, args.output_dir, sasrec_recall=sasrec_recall)
    print(f"Exported {len(frame)} points to {args.output_dir}")


if __name__ == "__main__":
    main()
