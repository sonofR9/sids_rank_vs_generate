#!/usr/bin/env python3
"""Plot every completed generation method from committed report tables.

The source reports are intentionally kept separate, but their CSV tables have
the same sweep coordinates and SID-quality columns. This utility joins their
completed runs into one poster-style two-panel figure without querying MLflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


LEVEL_SWEEP = (2, 3, 4)
CODEBOOK_SWEEP = (128, 256, 512, 1024)

BASE_METHOD_ORDER = (
    "R3VAE (rqkmeans)",
    "R3VAE (spherical)",
    "RQKMeans",
    "RQVAE (ECM)",
    "RQVAE",
    "RQVAE (RRS)",
    "RQVAE-token",
)
METHOD_ORDER = (*BASE_METHOD_ORDER, "PLUM", "BalanceRQKMeans", "RQVAE + item LogQ", "RQVAE + pure SID LogQ")
METHOD_STYLE = {
    "R3VAE (rqkmeans)": ("#72B7B2", "o", False),
    "R3VAE (spherical)": ("#54A24B", "o", False),
    "RQKMeans": ("#4C78A8", "o", False),
    "RQVAE (ECM)": ("#B279A2", "o", False),
    "RQVAE": ("#F58518", "o", False),
    "RQVAE (RRS)": ("#E45756", "D", False),
    "RQVAE-token": ("#FF9DA6", "s", True),
    "PLUM": ("#9D755D", "^", False),
    "BalanceRQKMeans": ("#7A5195", "s", False),
    "RQVAE + item LogQ": ("#8C6D31", "P", False),
    "RQVAE + pure SID LogQ": ("#E15759", "X", False),
}


def read_csv(path: Path, source: str) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Report table not found: {path}")
    frame = pd.read_csv(path)
    frame["source"] = source
    return frame


def first_present(frame: pd.DataFrame, *columns: str) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype=object)
    for column in columns:
        if column in frame:
            values = values.where(values.notna(), frame[column])
    return values


def normalize_base(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["method"] = first_present(frame, "method")
    output["num_levels"] = first_present(frame, "num_levels")
    output["codebook_size"] = first_present(frame, "codebook_size")
    output["recall_at_100"] = first_present(frame, "recall_at_100", "recall_100")
    output["code_entropy"] = first_present(frame, "code_entropy", "metrics.sid.code_entropy")
    output["run_name"] = first_present(frame, "run_name", "run_name_recall")
    output["source"] = frame["source"]
    return output


def normalize_logq(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    output = normalize_base(frame)
    output["method"] = method
    return output


def normalize_poster_extensions(frame: pd.DataFrame) -> pd.DataFrame:
    output = normalize_base(frame)
    output = output[output["method"].isin(("PLUM", "BalanceRQKMeans"))]
    # The historical report includes 12k-catalog points. New compatible runs
    # use catalog33148 in their names; exclude the historical points by default.
    return output[output["run_name"].astype(str).str.contains("catalog33148", na=False)]


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("num_levels", "codebook_size", "recall_at_100", "code_entropy"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["num_levels", "codebook_size", "recall_at_100", "code_entropy"])


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_key"] = list(zip(frame.method, frame.num_levels, frame.codebook_size, frame.run_name))
    return frame.drop_duplicates("_key", keep="last").drop(columns="_key")


def padded_limits(values: Iterable[float]) -> tuple[float, float]:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    low, high = float(clean.min()), float(clean.max())
    span = high - low
    padding = max(abs(low) * 0.04, 0.01) if span == 0 else span * 0.08
    return low - padding, high + padding


def draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    order_column: str,
    title: str,
    sweep_text: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    baseline: float | None,
) -> None:
    for method in METHOD_ORDER:
        points = frame[frame.method.eq(method)].sort_values(order_column)
        points = points.drop_duplicates(order_column)
        if points.empty:
            continue
        color, marker, filled = METHOD_STYLE[method]
        if len(points) > 1:
            ax.plot(points.code_entropy, points.recall_at_100, color=color, linewidth=1.25, alpha=0.72, zorder=1)
        ax.scatter(
            points.code_entropy,
            points.recall_at_100,
            s=58,
            marker=marker,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.55,
            zorder=3,
        )
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


def plot(frame: pd.DataFrame, output_dir: Path, baseline: float | None) -> None:
    left = frame[frame.codebook_size.eq(256) & frame.num_levels.isin(LEVEL_SWEEP)]
    right = frame[frame.num_levels.eq(3) & frame.codebook_size.isin(CODEBOOK_SWEEP)]
    shown = pd.concat([left, right], ignore_index=True).drop_duplicates(["method", "run_name"])
    if shown.empty:
        raise SystemExit("No points match the L/C poster sweeps")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 13,
        "axes.labelsize": 12, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 7.7, "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), sharey=True, facecolor="white")
    x_limits = padded_limits(shown.code_entropy)
    y_values = shown.recall_at_100 if baseline is None else pd.concat([shown.recall_at_100, pd.Series([baseline])])
    y_limits = padded_limits(y_values)
    draw_panel(axes[0], left, "num_levels", "Varying number of codebooks (C=256)", "L: 2 → 3 → 4", x_limits, y_limits, baseline)
    draw_panel(axes[1], right, "codebook_size", "Varying codebook size (L=3)", "C: 128 → 256 → 512 → 1024", x_limits, y_limits, baseline)
    figure.supxlabel("Code Entropy", fontsize=12, y=0.16)
    figure.supylabel("Recall@100", fontsize=12, x=0.016)

    present = [method for method in METHOD_ORDER if method in set(shown.method)]
    handles = []
    for method in present:
        color, marker, filled = METHOD_STYLE[method]
        handles.append(Line2D([0], [0], color=color, linewidth=1.1, marker=marker, markersize=6,
                              markerfacecolor=color if filled else "white", markeredgewidth=1.4, label=method))
    if baseline is not None:
        handles.append(Line2D([0], [0], color="#7B8794", linewidth=1.2, linestyle=(0, (5, 3)), label="SASRec"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4,
                  frameon=False, handlelength=1.8, columnspacing=1.15)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.38, wspace=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"all_methods_recall100_vs_code_entropy.{extension}", dpi=300 if extension == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", type=Path, default=Path("reports/figures/generation/raw_sid_recall/matched_runs.csv"))
    parser.add_argument("--item-logq-report", type=Path, default=Path("reports/figures/generation/rqvae_item_logq_recall/matched_logq_rqvae_runs.csv"))
    parser.add_argument("--pure-sid-logq-report", type=Path, default=Path("reports/figures/generation/rqvae_pure_sid_logq_recall/matched_logq_rqvae_runs.csv"))
    parser.add_argument("--poster-extensions-report", type=Path, default=Path("reports/figures/generation/poster_plum_balance_recall/matched_generation_runs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/generation/all_methods_recall"))
    parser.add_argument("--baseline-recall", type=float, default=0.113)
    args = parser.parse_args()

    raw = normalize_base(read_csv(args.raw_report, "raw_sid"))
    item_logq = normalize_logq(read_csv(args.item_logq_report, "item_logq"), "RQVAE + item LogQ")
    pure_sid_logq = normalize_logq(read_csv(args.pure_sid_logq_report, "pure_sid_logq"), "RQVAE + pure SID LogQ")
    extensions = normalize_poster_extensions(read_csv(args.poster_extensions_report, "poster_extensions"))
    frame = deduplicate(numeric(pd.concat([raw, item_logq, pure_sid_logq, extensions], ignore_index=True)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_methods_points.csv", index=False)
    plot(frame, args.output_dir, args.baseline_recall)
    print(f"Methods: {', '.join(method for method in METHOD_ORDER if method in set(frame.method))}")
    print(f"Points: {len(frame)}")
    print(f"Plots: {args.output_dir / 'all_methods_recall100_vs_code_entropy.png'}")


if __name__ == "__main__":
    main()
