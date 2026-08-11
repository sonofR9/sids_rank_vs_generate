#!/usr/bin/env python3
"""SID-quality correlations for generation runs using RQ-VAE with LogQ only.

Every accepted MLflow experiment is kept as a separate point. Plain RQ-VAE,
R3VAE, RQ-KMeans, baseline ``logq_mode=none`` runs, and other SID families are
explicitly rejected and recorded in CSV outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr


LEVEL_SWEEP = (2, 3, 4)
CODEBOOK_SWEEP = (128, 256, 512, 1024)
DISABLED_LOGQ_VALUES = {"", "none", "off", "false", "0", "baseline", "disabled"}


@dataclass(frozen=True)
class SidMetric:
    column: str
    label: str
    stem: str
    keys: tuple[str, ...]


SID_METRICS = (
    SidMetric(
        "icr",
        "Raw independent code rate (ICR)",
        "icr",
        (
            "analysis.raw_independent_code_rate",
            "sid.independent_code_rate",
            "assignment.base_icr",
        ),
    ),
    SidMetric("collision_rate", "Collision Rate", "collision_rate", ("sid.collision_rate",)),
    SidMetric(
        "p95_collision_group_size",
        "p95 Collision Group Size",
        "p95_collision_group_size",
        ("sid.collision_percentiles.p95",),
    ),
    SidMetric("g_max", "G_max", "g_max", ("sid.max_collision",)),
    SidMetric(
        "cur_total",
        "CUR_Total",
        "cur_total",
        ("sid.codebook_quality.codebook_utilization_rate_total",),
    ),
    SidMetric("code_gini", "Code Gini", "code_gini", ("sid.code_gini",)),
    SidMetric("code_entropy", "Code Entropy", "code_entropy", ("sid.code_entropy",)),
    SidMetric(
        "intra_code_similarity",
        "Intra-code Similarity",
        "intra_code_similarity",
        ("sid.intra_code_similarity",),
    ),
    SidMetric(
        "prefix_coherence_1",
        "Prefix Coherence@1",
        "prefix_coherence_at_1",
        ("sid.prefix_coherence_at_1",),
    ),
    SidMetric(
        "prefix_coherence_2",
        "Prefix Coherence@2",
        "prefix_coherence_at_2",
        ("sid.prefix_coherence_at_2",),
    ),
    SidMetric(
        "prefix_coherence_3",
        "Prefix Coherence@3",
        "prefix_coherence_at_3",
        ("sid.prefix_coherence_at_3",),
    ),
    SidMetric("valid_sid_rate", "Valid SID Rate", "valid_sid_rate", ("sid.valid_sid_rate",)),
)

RECALL_KEYS = (
    "evaluations.generation.recall_at_100",
    "evaluations.generation.recall@100",
    "generation.recall_at_100",
    "generation.recall@100",
)

LOGQ_MODE_COLUMNS = (
    "tags.logq_mode",
    "params.training.generation.logq_mode",
    "params.config.logq_mode",
    "params.generation.logq_mode",
    "params.logq_mode",
)
RQ_VAE_COLUMNS = (
    "tags.sid_model",
    "params.sid_model",
    "params.config.sid_model",
    "tags.model",
    "params.model",
)

MARKERS = ("o", "s", "D", "^", "v", "P", "X", "<", ">", "h")
SERIES_COLORS = (
    "#F58518",  # RQ-VAE orange, matching the ranking figures.
    "#4C78A8",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
)

MATCHED_COLUMNS = (
    "model",
    "init",
    "assignment",
    "collision_resolution",
    "num_levels",
    "codebook_size",
    "seed",
    "dataset",
    "data_scope",
    "run_name",
    "logq_run_id",
    "experiment_name",
    "status",
    "start_time",
    "recall_at_100",
    "logq_mode",
    "logq_signature",
    "rqvae_filter_evidence",
    "logq_filter_evidence",
    "generation_child_run_ids",
    "sid_quality_run_id",
    "sid_quality_experiment",
    "quality_backfilled_metrics",
    *(spec.column for spec in SID_METRICS),
)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def text(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return str(value).strip()


def first(row: pd.Series, *columns: str, default: Any = None) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]) and text(row[column]):
            return row[column]
    return default


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def split_csv(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(result))


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.Series(dtype=object)
    return result


def resolve_metric(columns: Iterable[str], keys: Iterable[str]) -> str | None:
    metric_columns = [column for column in columns if column.startswith("metrics.")]
    for key in keys:
        exact = f"metrics.{key}"
        if exact in metric_columns:
            return exact
    normalized_columns = {
        normalize(column.removeprefix("metrics.")): column for column in metric_columns
    }
    for key in keys:
        if normalize(key) in normalized_columns:
            return normalized_columns[normalize(key)]
    return None


def load_runs(experiment_names: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    experiment_ids: dict[str, str] = {}
    for name in experiment_names:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            raise SystemExit(f"MLflow experiment not found: {name}")
        experiment_ids[name] = experiment.experiment_id
        frame = mlflow.search_runs(
            [experiment.experiment_id], max_results=10000, output_format="pandas"
        )
        frame["experiment_name"] = name
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True, sort=False)
    return runs.drop_duplicates("run_id", keep="last").reset_index(drop=True), experiment_ids


def inject_generation_child_recall(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = runs.copy()
    parent_column = "tags.mlflow.parentRunId"
    target = "metrics.evaluations.generation.recall_at_100"
    if target not in output:
        output[target] = np.nan
    if "generation_child_run_ids" not in output:
        output["generation_child_run_ids"] = ""
    if parent_column not in output:
        return output, pd.DataFrame()

    parent_indices = {
        text(run_id): index for index, run_id in output["run_id"].items() if text(run_id)
    }
    evidence = []
    for _, child in output[output[parent_column].notna()].iterrows():
        parent_id = text(child.get(parent_column))
        parent_index = parent_indices.get(parent_id)
        if parent_index is None:
            continue
        name = text(first(child, "tags.mlflow.runName", "run_name")).lower()
        stage = text(first(child, "tags.stage")).lower()
        if stage != "generation" and "generation" not in name:
            continue
        source_candidates = (
            "metrics.test.recall_at_100",
            "metrics.test.recall@100",
            "metrics.recall_at_100",
            "metrics.recall@100",
            target,
        )
        source = ""
        value = None
        for candidate in source_candidates:
            if candidate in child and finite(child.get(candidate)):
                source = candidate
                value = float(child[candidate])
                break
        if value is None:
            continue
        if not finite(output.at[parent_index, target]):
            output.at[parent_index, target] = value
        existing = [
            item for item in text(output.at[parent_index, "generation_child_run_ids"]).split(";") if item
        ]
        child_id = text(child.get("run_id"))
        if child_id not in existing:
            existing.append(child_id)
        output.at[parent_index, "generation_child_run_ids"] = ";".join(existing)
        evidence.append(
            {
                "parent_run_id": parent_id,
                "child_run_id": child_id,
                "child_run_name": text(first(child, "tags.mlflow.runName", "run_name")),
                "child_status": text(child.get("status")),
                "source_key": source.removeprefix("metrics."),
                "target_key": target.removeprefix("metrics."),
                "recall_at_100": value,
            }
        )
    return output, pd.DataFrame(evidence)


def parent_runs(runs: pd.DataFrame) -> pd.DataFrame:
    result = runs.copy()
    if "tags.mlflow.parentRunId" in result:
        result = result[result["tags.mlflow.parentRunId"].isna()]
    return result.reset_index(drop=True)


def parse_config(row: pd.Series) -> dict[str, Any]:
    run_name = text(first(row, "tags.mlflow.runName", "run_name"))
    lower_name = run_name.lower()
    model = text(first(row, "tags.sid_model", "params.sid_model", "tags.model")).lower()
    init = text(first(row, "tags.init", "params.config.init", "params.init"), "rqkmeans").lower()
    assignment = text(
        first(row, "tags.assignment", "params.config.assignment", "params.assignment"),
        "nearest",
    ).lower()
    collision = text(
        first(
            row,
            "tags.collision_resolution",
            "params.config.collision_resolution",
            "params.collision_resolution",
        ),
        "none",
    ).lower()

    pattern = re.search(
        r"(?:logq-target-|poster(?:-ranking)?-|sid-quality-)"
        r"(?P<model>rqvae|rq-vae|r3vae|rqkmeans)-"
        r"(?P<init>rqkmeans|spherical)-"
        r"(?P<assignment>nearest|rrs|ecm)"
        r"(?:-(?P<collision>none|token))?",
        lower_name,
    )
    if pattern:
        model = model or pattern.group("model").replace("-", "")
        init = init or pattern.group("init")
        assignment = assignment or pattern.group("assignment")
        if pattern.group("collision"):
            collision = pattern.group("collision")

    levels = first(row, "tags.num_levels", "params.config.num_levels", "params.num_levels")
    if levels is None:
        match = re.search(r"(?:^|-)l(\d+)(?:-|$)", lower_name)
        levels = match.group(1) if match else None
    codebook = first(
        row, "tags.codebook_size", "params.config.codebook_size", "params.codebook_size"
    )
    if codebook is None:
        match = re.search(r"(?:^|-)c(\d+)(?:-|$)", lower_name)
        codebook = match.group(1) if match else None
    seed = first(row, "tags.seed", "params.config.seed", "params.seed", default=42)
    dataset = text(
        first(row, "tags.dataset", "params.dataset.size", "params.dataset", "params.config.dataset"),
        "50m",
    ).lower()
    if dataset.isdigit():
        dataset = f"{dataset}m"
    scope = text(first(row, "tags.data_scope", "params.data_scope"), "full").lower()
    return {
        "model": model.replace("-", ""),
        "init": init,
        "assignment": assignment,
        "collision_resolution": collision,
        "num_levels": to_int(levels),
        "codebook_size": to_int(codebook),
        "seed": to_int(seed, 42),
        "dataset": dataset,
        "data_scope": scope,
        "run_name": run_name,
    }


def rqvae_evidence(row: pd.Series, config: dict[str, Any]) -> tuple[bool, str]:
    explicit = []
    for column in RQ_VAE_COLUMNS:
        value = text(row.get(column)).lower().replace("-", "").replace("_", "")
        if value:
            explicit.append(f"{column}={value}")
    if explicit:
        accepted = any("rqvae" == item.split("=", 1)[1] for item in explicit)
        return accepted, ";".join(explicit)
    name = config["run_name"].lower().replace("-", "").replace("_", "")
    return "rqvae" in name and "r3vae" not in name, "run_name_fallback"


def logq_evidence(row: pd.Series, run_name: str) -> tuple[bool, str, str]:
    values = []
    for column in LOGQ_MODE_COLUMNS:
        value = text(row.get(column)).lower()
        if value:
            values.append((column, value))
    if values:
        column, mode = values[0]
        return mode not in DISABLED_LOGQ_VALUES, mode, f"{column}={mode}"

    target = text(row.get("tags.logq_target")).lower()
    name = run_name.lower()
    if "corrected" in name:
        return True, "corrected", "run_name=corrected"
    if "standard" in name:
        return True, "standard", "run_name=standard"
    if target in {"true", "1", "yes"} and "baseline" not in name:
        return True, "enabled", "tags.logq_target=true"
    return False, "none", "no enabled LogQ evidence"


def logq_signature(row: pd.Series, mode: str) -> str:
    signature: dict[str, str] = {"logq_mode": mode}
    for column, value in row.items():
        lower = column.lower()
        if not (column.startswith("tags.") or column.startswith("params.")):
            continue
        if "logq" in lower or any(
            token in lower
            for token in (
                "auxiliary_item_loss_weight",
                "ranking_temperature",
                "gather_distributed_negatives",
            )
        ):
            rendered = text(value)
            if rendered:
                signature[column] = rendered
    return json.dumps(signature, sort_keys=True, ensure_ascii=False)


def collect_quality_runs(
    runs: pd.DataFrame, sid_mapping: dict[str, str | None]
) -> pd.DataFrame:
    records = []
    for _, row in parent_runs(runs).iterrows():
        config = parse_config(row)
        is_rqvae, _ = rqvae_evidence(row, config)
        if not is_rqvae:
            continue
        record = {
            **config,
            "quality_run_id": text(row.get("run_id")),
            "quality_experiment": text(row.get("experiment_name")),
            "quality_start_time": text(row.get("start_time")),
        }
        for spec in SID_METRICS:
            column = sid_mapping.get(spec.column)
            record[spec.column] = row.get(column, np.nan) if column else np.nan
        if any(finite(record[spec.column]) for spec in SID_METRICS):
            records.append(record)
    return pd.DataFrame(records)


def collect_logq_runs(
    runs: pd.DataFrame,
    recall_column: str | None,
    sid_mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = []
    rejected = []
    for _, row in parent_runs(runs).iterrows():
        recall = row.get(recall_column, np.nan) if recall_column else np.nan
        if not finite(recall):
            continue
        config = parse_config(row)
        is_rqvae, rqvae_reason = rqvae_evidence(row, config)
        has_logq, mode, logq_reason = logq_evidence(row, config["run_name"])
        base = {
            **config,
            "logq_run_id": text(row.get("run_id")),
            "experiment_name": text(row.get("experiment_name")),
            "status": text(row.get("status")),
            "start_time": text(row.get("start_time")),
            "recall_at_100": float(recall),
            "logq_mode": mode,
            "rqvae_filter_evidence": rqvae_reason,
            "logq_filter_evidence": logq_reason,
            "generation_child_run_ids": text(row.get("generation_child_run_ids")),
        }
        if not is_rqvae:
            rejected.append({**base, "rejection_reason": "not RQ-VAE"})
            continue
        if not has_logq:
            rejected.append({**base, "rejection_reason": "LogQ disabled or unproven"})
            continue
        base["logq_signature"] = logq_signature(row, mode)
        for spec in SID_METRICS:
            column = sid_mapping.get(spec.column)
            base[spec.column] = row.get(column, np.nan) if column else np.nan
        accepted.append(base)
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def quality_match_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        row.get("model"),
        row.get("init"),
        row.get("assignment"),
        row.get("num_levels"),
        row.get("codebook_size"),
        row.get("seed"),
        row.get("dataset"),
        row.get("data_scope"),
    )


def match_quality(
    logq_runs: pd.DataFrame, quality_runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if logq_runs.empty:
        return logq_runs.copy(), logq_runs.copy()
    quality = quality_runs.copy()
    quality_lookup: dict[tuple[Any, ...], pd.Series] = {}
    if not quality.empty:
        quality["_key"] = quality.apply(quality_match_key, axis=1)
        metric_columns = [spec.column for spec in SID_METRICS]
        quality["_coverage"] = quality[metric_columns].apply(
            lambda row: sum(finite(value) for value in row), axis=1
        )
        quality = quality.sort_values(["_coverage", "quality_start_time"])
        quality = quality.drop_duplicates("_key", keep="last")
        quality_lookup = {
            row["_key"]: row for _, row in quality.iterrows()
        }

    matched = []
    unmatched = []
    for _, run in logq_runs.iterrows():
        output = run.to_dict()
        key = quality_match_key(run)
        quality_row = quality_lookup.get(key)
        filled = []
        if quality_row is not None:
            output["sid_quality_run_id"] = quality_row["quality_run_id"]
            output["sid_quality_experiment"] = quality_row["quality_experiment"]
            for spec in SID_METRICS:
                if not finite(output.get(spec.column)) and finite(quality_row.get(spec.column)):
                    output[spec.column] = float(quality_row[spec.column])
                    filled.append(spec.column)
        else:
            output["sid_quality_run_id"] = ""
            output["sid_quality_experiment"] = ""
        output["quality_backfilled_metrics"] = ";".join(filled)
        if any(finite(output.get(spec.column)) for spec in SID_METRICS):
            matched.append(output)
        else:
            unmatched.append({**output, "unmatched_reason": "no SID quality metrics"})
    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def point_label(row: pd.Series) -> str:
    name = text(row.get("run_name"))
    if len(name) <= 72:
        return name
    return f"{name[:58]}… [{text(row.get('logq_run_id'))[:8]}]"


def point_styles(matched: pd.DataFrame) -> dict[str, tuple[Any, str]]:
    ordered = matched.sort_values(
        ["logq_mode", "logq_signature", "run_name", "logq_run_id"]
    )
    signatures = list(dict.fromkeys(ordered["logq_signature"].tolist()))
    signature_colors = {
        signature: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, signature in enumerate(signatures)
    }
    styles: dict[str, tuple[Any, str]] = {}
    for _, group in ordered.groupby("logq_signature", sort=False, dropna=False):
        for _, row in group.iterrows():
            styles[row.logq_run_id] = (
                signature_colors[row.logq_signature],
                "o",
            )
    return styles


def series_labels(frame: pd.DataFrame) -> dict[str, str]:
    """Build compact method labels while keeping LogQ signatures separate."""
    representatives = (
        frame.sort_values(["logq_mode", "logq_signature", "run_name"])
        .drop_duplicates("logq_signature")
    )
    labels: dict[str, str] = {}
    used: dict[str, int] = {}
    for _, row in representatives.iterrows():
        mode = text(row.get("logq_mode"), "enabled")
        label = "RQ-VAE + LogQ"
        if mode not in {"", "enabled"}:
            label += f" ({mode})"
        try:
            signature = json.loads(text(row.get("logq_signature"), "{}"))
        except json.JSONDecodeError:
            signature = {}
        if len(representatives) > 1:
            aux_values = [
                text(value)
                for key, value in signature.items()
                if "auxiliary_item_loss_weight" in key.lower() and text(value)
            ]
            if aux_values:
                label += f", aux={aux_values[0]}"
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label += f" — variant {used[label]}"
        labels[row.logq_signature] = label
    return labels


def padded_limits(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 1.0
    low, high = float(clean.min()), float(clean.max())
    span = high - low
    padding = max(abs(low) * 0.04, 0.01) if span == 0 else span * 0.08
    return low - padding, high + padding


def draw_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    sid_metric: SidMetric,
    styles: dict[str, tuple[Any, str]],
    order_column: str,
    title: str,
    sweep_text: str,
    y_limits: tuple[float, float],
) -> None:
    # Connect only runs with exactly the same LogQ signature. Individual runs
    # are never averaged or collapsed.
    for _, group in frame.groupby("logq_signature", dropna=False):
        group = group.sort_values([order_column, "start_time"])
        if len(group) > 1:
            color = styles[group.iloc[0].logq_run_id][0]
            axis.plot(
                group[sid_metric.column],
                group["recall_at_100"],
                color=color,
                linewidth=1.1,
                alpha=0.5,
                zorder=1,
            )
    for _, row in frame.iterrows():
        color, marker = styles[row.logq_run_id]
        axis.scatter(
            [row[sid_metric.column]],
            [row.recall_at_100],
            s=62,
            marker=marker,
            facecolors="white",
            edgecolors=[color],
            linewidths=1.65,
            zorder=3,
        )
    axis.set_title(title, pad=9, fontweight="normal")
    axis.set_ylim(*y_limits)
    axis.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.42)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#94A3B8")
    axis.spines["bottom"].set_color("#94A3B8")
    axis.tick_params(colors="#334155", width=0.8)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7))
    axis.text(
        0.5,
        -0.18,
        sweep_text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#64748B",
        clip_on=False,
    )


def plot_metric(
    matched: pd.DataFrame,
    sid_metric: SidMetric,
    styles: dict[str, tuple[Any, str]],
    output_dir: Path,
) -> bool:
    frame = matched.copy()
    frame[sid_metric.column] = pd.to_numeric(frame[sid_metric.column], errors="coerce")
    frame["recall_at_100"] = pd.to_numeric(frame["recall_at_100"], errors="coerce")
    left = frame[
        frame.codebook_size.eq(256) & frame.num_levels.isin(LEVEL_SWEEP)
    ].dropna(subset=[sid_metric.column, "recall_at_100"])
    right = frame[
        frame.num_levels.eq(3) & frame.codebook_size.isin(CODEBOOK_SWEEP)
    ].dropna(subset=[sid_metric.column, "recall_at_100"])
    shown = pd.concat([left, right], ignore_index=True).drop_duplicates("logq_run_id")
    if shown.empty:
        return False

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
        facecolor="white",
    )
    x_limits = padded_limits(shown[sid_metric.column])
    y_limits = padded_limits(shown["recall_at_100"])
    for axis in axes:
        axis.set_facecolor("white")
        axis.set_xlim(*x_limits)
    draw_panel(
        axes[0],
        left,
        sid_metric,
        styles,
        "num_levels",
        "Varying number of codebooks (C=256)",
        "L: 2 → 3 → 4",
        y_limits,
    )
    draw_panel(
        axes[1],
        right,
        sid_metric,
        styles,
        "codebook_size",
        "Varying codebook size (L=3)",
        "C: 128 → 256 → 512 → 1024",
        y_limits,
    )
    figure.supxlabel(sid_metric.label, fontsize=12, y=0.13)
    figure.supylabel("Recall@100", fontsize=12, x=0.016)

    labels = series_labels(shown)
    representatives = (
        shown.sort_values(["logq_mode", "logq_signature", "run_name"])
        .drop_duplicates("logq_signature")
    )
    handles = []
    for _, row in representatives.iterrows():
        color, marker = styles[row.logq_run_id]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.15,
                marker=marker,
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                label=labels[row.logq_signature],
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=min(3, max(1, len(handles))),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.35,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.31,
        wspace=0.06,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"recall100_logq_rqvae_vs_{sid_metric.stem}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return True


def correlation_table(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", matched)]
    groups.extend((f"logq_mode={mode}", part) for mode, part in matched.groupby("logq_mode"))
    for spec in SID_METRICS:
        for group_name, part in groups:
            pairs = (
                pd.DataFrame(
                    {
                        "x": pd.to_numeric(part[spec.column], errors="coerce"),
                        "y": pd.to_numeric(part["recall_at_100"], errors="coerce"),
                    }
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            pearson = np.nan
            spearman = np.nan
            if len(pairs) >= 2 and pairs.x.nunique() >= 2 and pairs.y.nunique() >= 2:
                pearson = float(pearsonr(pairs.x, pairs.y).statistic)
                spearman = float(spearmanr(pairs.x, pairs.y).statistic)
            rows.append(
                {
                    "downstream_metric": "recall_at_100",
                    "sid_metric": spec.column,
                    "sid_metric_label": spec.label,
                    "group": group_name,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "num_points": len(pairs),
                }
            )
    return pd.DataFrame(rows)


def metric_coverage(
    matched: pd.DataFrame, sid_mapping: dict[str, str | None], recall_column: str | None
) -> pd.DataFrame:
    rows = []
    for spec in SID_METRICS:
        values = pd.to_numeric(matched.get(spec.column, pd.Series(dtype=float)), errors="coerce")
        finite_count = int(values.replace([np.inf, -np.inf], np.nan).notna().sum())
        resolved = sid_mapping.get(spec.column)
        rows.append(
            {
                "metric": spec.column,
                "requested_keys": ";".join(spec.keys),
                "resolved_key": resolved.removeprefix("metrics.") if resolved else "",
                "num_logq_runs": len(matched),
                "finite_values": finite_count,
                "missing_values": len(matched) - finite_count,
            }
        )
    recall_values = pd.to_numeric(
        matched["recall_at_100"]
        if "recall_at_100" in matched
        else pd.Series(index=matched.index, dtype=float),
        errors="coerce",
    )
    rows.append(
        {
            "metric": "recall_at_100",
            "requested_keys": ";".join(RECALL_KEYS),
            "resolved_key": recall_column.removeprefix("metrics.") if recall_column else "",
            "num_logq_runs": len(matched),
            "finite_values": int(recall_values.notna().sum()),
            "missing_values": int(recall_values.isna().sum()),
        }
    )
    return pd.DataFrame(rows)


def git_value(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument(
        "--logq-experiments",
        action="append",
        default=[],
        help="Comma-separated MLflow experiments containing generation runs.",
    )
    parser.add_argument(
        "--quality-experiments",
        action="append",
        default=[],
        help="Optional quality-only experiments used to backfill SID metrics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    logq_experiments = split_csv(args.logq_experiments) or ["sid-comparison", "test"]
    quality_experiments = split_csv(args.quality_experiments) or ["sid-quality-only-full-v2"]
    experiment_names = list(dict.fromkeys([*logq_experiments, *quality_experiments]))

    mlflow.set_tracking_uri(args.tracking_uri)
    runs, experiment_ids = load_runs(experiment_names)
    runs, child_evidence = inject_generation_child_recall(runs)
    metric_columns = [column for column in runs.columns if column.startswith("metrics.")]
    recall_column = resolve_metric(metric_columns, RECALL_KEYS)
    sid_mapping = {spec.column: resolve_metric(metric_columns, spec.keys) for spec in SID_METRICS}

    print("Resolved MLflow metrics:")
    print(
        "  Recall@100 ->",
        (recall_column or "<missing>").removeprefix("metrics."),
    )
    for spec in SID_METRICS:
        resolved = sid_mapping[spec.column]
        print(f"  {spec.label:<34} -> {(resolved or '<missing>').removeprefix('metrics.')}")

    quality_runs = collect_quality_runs(runs, sid_mapping)
    logq_runs, rejected = collect_logq_runs(runs, recall_column, sid_mapping)
    matched, unmatched = match_quality(logq_runs, quality_runs)
    matched = ensure_columns(matched, MATCHED_COLUMNS)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / "matched_logq_rqvae_runs.csv", index=False)
    rejected.to_csv(output_dir / "rejected_non_logq_or_non_rqvae_runs.csv", index=False)
    unmatched.to_csv(output_dir / "unmatched_logq_rqvae_runs.csv", index=False)
    child_evidence.to_csv(output_dir / "generation_child_recall_sources.csv", index=False)

    coverage = metric_coverage(matched, sid_mapping, recall_column)
    coverage.to_csv(output_dir / "metric_coverage_summary.csv", index=False)
    correlations = correlation_table(matched)
    correlations.to_csv(output_dir / "correlations.csv", index=False)

    outside = matched[
        ~(
            (matched.codebook_size.eq(256) & matched.num_levels.isin(LEVEL_SWEEP))
            | (matched.num_levels.eq(3) & matched.codebook_size.isin(CODEBOOK_SWEEP))
        )
    ].copy()
    outside.to_csv(output_dir / "outside_standard_plot_sweeps.csv", index=False)

    matched = matched.copy()
    matched["point_label"] = matched.apply(point_label, axis=1)
    matched[
        [
            "point_label",
            "logq_run_id",
            "run_name",
            "logq_mode",
            "logq_signature",
            "num_levels",
            "codebook_size",
            "recall_at_100",
        ]
    ].to_csv(output_dir / "point_legend.csv", index=False)

    styles = point_styles(matched) if not matched.empty else {}
    plotted = []
    for spec in SID_METRICS:
        if plot_metric(matched, spec, styles, output_dir / "plots"):
            plotted.append(spec.column)

    metadata = {
        "tracking_uri": args.tracking_uri,
        "experiments": experiment_ids,
        "logq_experiments": logq_experiments,
        "quality_experiments": quality_experiments,
        "filters": {
            "sid_model": "RQ-VAE only",
            "logq": "explicit enabled mode only; none/baseline rejected",
            "one_point_per_mlflow_run": True,
            "plain_rqvae_excluded": True,
            "other_sid_families_excluded": True,
        },
        "metric_mapping": {
            "recall_at_100": recall_column,
            "sid": sid_mapping,
        },
        "rqvae_logq_runs_found": len(logq_runs),
        "matched_runs": len(matched),
        "unmatched_runs": len(unmatched),
        "rejected_runs": len(rejected),
        "logq_mode_counts": matched.logq_mode.value_counts().to_dict() if not matched.empty else {},
        "plotted_sid_metrics": plotted,
        "generation_child_recall_rows": len(child_evidence),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "git_branch": git_value(Path(__file__).resolve().parents[1], "branch", "--show-current"),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nData integrity summary:")
    print(f"  RQ-VAE + LogQ runs found: {len(logq_runs)}")
    print(f"  Matched with SID metrics:  {len(matched)}")
    print(f"  Unmatched LogQ runs:       {len(unmatched)}")
    print(f"  Rejected recall runs:      {len(rejected)}")
    print(f"  Outside standard sweeps:   {len(outside)}")
    if not matched.empty:
        print("  LogQ modes:")
        for mode, count in matched.logq_mode.value_counts().items():
            print(f"    {mode:<18} {count}")
        print("  Accepted runs:")
        for run_name in matched.run_name.tolist():
            print(f"    {run_name}")
    if not rejected.empty:
        print("  Rejection reasons:")
        for reason, count in rejected.rejection_reason.value_counts().items():
            print(f"    {reason:<30} {count}")
    print("\nMetric coverage:")
    print(coverage.to_string(index=False))
    print(f"\nPlots written per format: {len(plotted)}")


if __name__ == "__main__":
    main()
