#!/usr/bin/env python3
"""Export SID-quality versus generation Recall@100 figures from MLflow.

The exporter is read-only. It accepts both flattened parent metrics and the
nested ``generation`` child-run layout used by older experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mlflow.tracking import MlflowClient
from scipy.stats import pearsonr, spearmanr


LEVEL_SWEEP = (2, 3, 4)
CODEBOOK_SWEEP = (128, 256, 512, 1024)
RECALL_KEYS = (
    "evaluations.generation.recall_at_100",
    "evaluations.generation.recall@100",
    "generation.recall_at_100",
    "generation.recall@100",
)


@dataclass(frozen=True)
class MetricSpec:
    requested_key: str
    column: str
    label: str
    stem: str
    aliases: tuple[str, ...] = ()


SID_METRICS = (
    MetricSpec(
        "sid.independent_code_rate",
        "icr",
        "Raw independent code rate (ICR)",
        "icr",
        (
            "analysis.raw_independent_code_rate",
            "assignment.base_icr",
            "sid/sid_statistics/ICR",
        ),
    ),
    MetricSpec(
        "sid.collision_rate",
        "collision_rate",
        "Collision Rate",
        "collision_rate",
        ("sid/sid_statistics/Collisions_rate",),
    ),
    MetricSpec(
        "sid.collision_percentiles.p95",
        "p95_collision_group_size",
        "p95 Collision Group Size",
        "p95_collision_group_size",
    ),
    MetricSpec(
        "sid.max_collision",
        "g_max",
        "G_max",
        "g_max",
        ("sid/sid_statistics/max_collision",),
    ),
    MetricSpec(
        "sid.codebook_quality.codebook_utilization_rate_total",
        "cur_total",
        "CUR_Total",
        "cur_total",
        ("sid/codebook/codebook_utilization_rate_total",),
    ),
    MetricSpec("sid.code_gini", "code_gini", "Code Gini", "code_gini"),
    MetricSpec("sid.code_entropy", "code_entropy", "Code Entropy", "code_entropy"),
    MetricSpec(
        "sid.intra_code_similarity",
        "intra_code_similarity",
        "Intra-code Similarity",
        "intra_code_similarity",
    ),
    MetricSpec(
        "sid.prefix_coherence_at_1",
        "prefix_coherence_1",
        "Prefix Coherence@1",
        "prefix_coherence_at_1",
    ),
    MetricSpec(
        "sid.prefix_coherence_at_2",
        "prefix_coherence_2",
        "Prefix Coherence@2",
        "prefix_coherence_at_2",
    ),
    MetricSpec(
        "sid.prefix_coherence_at_3",
        "prefix_coherence_3",
        "Prefix Coherence@3",
        "prefix_coherence_at_3",
    ),
    MetricSpec("sid.valid_sid_rate", "valid_sid_rate", "Valid SID Rate", "valid_sid_rate"),
)

METHOD_ORDER = (
    "R3VAE (rqkmeans)",
    "R3VAE (spherical)",
    "RQKMeans",
    "RQ-KMeans + Pair Alignment",
    "RQ-VAE + Pair Alignment",
    "RQVAE (ECM)",
    "RQVAE",
    "RQVAE (RRS)",
    "PLUM",
    "BalanceRQKMeans",
    "RQVAE-token",
)
METHOD_STYLE = {
    "R3VAE (rqkmeans)": ("#72B7B2", "o", False),
    "R3VAE (spherical)": ("#54A24B", "o", False),
    "RQKMeans": ("#4C78A8", "o", False),
    "RQ-KMeans + Pair Alignment": ("#2A9D8F", "P", True),
    "RQ-VAE + Pair Alignment": ("#E9C46A", "X", True),
    "RQVAE (ECM)": ("#B279A2", "o", False),
    "RQVAE": ("#F58518", "o", False),
    "RQVAE (RRS)": ("#E45756", "D", False),
    "PLUM": ("#9D755D", "^", False),
    "BalanceRQKMeans": ("#7A5195", "s", False),
    "RQVAE-token": ("#FF9DA6", "s", True),
}


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


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def split_csv(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(result))


def resolve_metric(columns: Iterable[str], keys: Iterable[str]) -> str | None:
    metric_columns = [column for column in columns if column.startswith("metrics.")]
    for key in keys:
        exact = f"metrics.{key}"
        if exact in metric_columns:
            return exact
    normalized = {
        normalize_key(column.removeprefix("metrics.")): column
        for column in metric_columns
    }
    for key in keys:
        if normalize_key(key) in normalized:
            return normalized[normalize_key(key)]
    return None


def parse_run_config(row: pd.Series) -> dict[str, Any]:
    name = text(first(row, "tags.mlflow.runName", "run_name"))
    lower_name = name.lower()
    family = text(first(row, "tags.sid_model", "params.sid_model")).lower()
    if (
        "onerec-pair-rqvae" in lower_name
        or family == "collaborative_rqvae"
    ):
        family = "onerec_pair_rqvae"
    elif (
        "onerec-pair" in lower_name
        or family in {"collaborative_rqkmeans", "onerec_pair"}
    ):
        family = "onerec_pair"
    elif "plum" in lower_name or family == "plum":
        family = "plum"
    elif "balance" in lower_name or family in {"balance", "balance_rqkmeans", "balancerqkmeans"}:
        family = "balance"
    else:
        family = "poster"

    model = text(first(row, "tags.model", "params.model", "params.config.model")).lower()
    init = text(first(row, "tags.init", "params.init", "params.config.init")).lower()
    assignment = text(first(row, "tags.assignment", "params.assignment", "params.config.assignment")).lower()
    collision = text(
        first(row, "tags.collision_resolution", "params.collision_resolution"),
        "none",
    ).lower()
    match = re.search(
        r"(?:poster(?:-ranking)?|sid-quality|logq-target)-"
        r"(?P<model>rqkmeans|rqvae|r3vae)-"
        r"(?P<init>rqkmeans|spherical)-"
        r"(?P<assignment>nearest|rrs|ecm)"
        r"(?:-(?P<collision>none|token))?",
        lower_name,
    )
    if match:
        model = model or match.group("model")
        init = init or match.group("init")
        assignment = assignment or match.group("assignment")
        collision = match.group("collision") or collision

    if family == "onerec_pair_rqvae":
        model, init, assignment = "collaborative_rqvae", "rqkmeans", "nearest"
    elif family == "onerec_pair":
        model, init, assignment = "collaborative_rqkmeans", "rqkmeans", "nearest"
    elif family == "plum":
        model, init, assignment = "plum", "plum", "nearest"
    elif family == "balance":
        model, init, assignment = "balance", "rqkmeans", "nearest"

    levels = first(row, "tags.num_levels", "params.num_levels", "params.config.num_levels")
    if levels is None:
        found = re.search(r"(?:^|-)l(\d+)(?:-|$)", lower_name)
        levels = found.group(1) if found else None
    codebook = first(row, "tags.codebook_size", "params.codebook_size", "params.config.codebook_size")
    if codebook is None:
        found = re.search(r"(?:^|-)c(\d+)(?:eq)?(?:-|$)", lower_name)
        codebook = found.group(1) if found else None
    sizes = text(first(row, "tags.codebook_sizes", "params.codebook_sizes"))
    sizes = sizes.lower().replace(",", "x").replace(" ", "")
    if not sizes:
        found = re.search(r"(?:^|-)cbs([0-9x]+)(?:-|$)", lower_name)
        sizes = found.group(1) if found else text(codebook)

    seed = first(row, "tags.seed", "params.seed", "params.config.seed", default=42)
    dataset = text(first(row, "tags.dataset", "params.dataset", "params.dataset.size"), "50m").lower()
    if dataset.isdigit():
        dataset = f"{dataset}m"
    scope = text(first(row, "tags.data_scope", "params.data_scope"), "full").lower()

    if family == "onerec_pair_rqvae":
        method = "RQ-VAE + Pair Alignment"
    elif family == "onerec_pair":
        method = "RQ-KMeans + Pair Alignment"
    elif family == "plum":
        method = "PLUM"
    elif family == "balance":
        method = "BalanceRQKMeans"
    elif collision == "token":
        method = "RQVAE-token"
    elif model == "rqkmeans":
        method = "RQKMeans"
    elif model == "rqvae" and assignment == "rrs":
        method = "RQVAE (RRS)"
    elif model == "rqvae" and assignment == "ecm":
        method = "RQVAE (ECM)"
    elif model == "rqvae":
        method = "RQVAE"
    elif model == "r3vae" and init == "spherical":
        method = "R3VAE (spherical)"
    elif model == "r3vae":
        method = "R3VAE (rqkmeans)"
    else:
        method = model or family or "unknown"

    return {
        "sid_family": family,
        "model": model,
        "init": init,
        "assignment": assignment,
        "collision_resolution": collision,
        "num_levels": to_int(levels),
        "codebook_size": to_int(codebook),
        "codebook_sizes": sizes,
        "seed": to_int(seed, 42),
        "dataset": dataset,
        "data_scope": scope,
        "method": method,
        "run_name": name,
    }


def load_runs(experiments: list[str]) -> pd.DataFrame:
    frames = []
    for name in experiments:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            raise SystemExit(f"MLflow experiment not found: {name}")
        frame = mlflow.search_runs([experiment.experiment_id], max_results=10000, output_format="pandas")
        frame["experiment_name"] = name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("run_id", keep="last")


def inject_generation_recall(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = runs.copy()
    parent_column = "tags.mlflow.parentRunId"
    target = "metrics.evaluations.generation.recall_at_100"
    if target not in output:
        output[target] = np.nan
    if parent_column not in output:
        return output, pd.DataFrame()

    parents = {text(run_id): index for index, run_id in output["run_id"].items() if text(run_id)}
    evidence = []
    candidates = (
        "metrics.test.recall_at_100",
        "metrics.test.recall@100",
        "metrics.recall_at_100",
        "metrics.recall@100",
        target,
    )
    for _, child in output[output[parent_column].notna()].iterrows():
        parent_id = text(child.get(parent_column))
        parent_index = parents.get(parent_id)
        if parent_index is None:
            continue
        name = text(first(child, "tags.mlflow.runName", "run_name")).lower()
        stage = text(first(child, "tags.stage")).lower()
        if stage and stage != "generation" and "generation" not in name:
            continue
        value = None
        source = ""
        for candidate in candidates:
            if candidate in child and finite(child.get(candidate)):
                value = float(child[candidate])
                source = candidate.removeprefix("metrics.")
                break
        if value is None:
            continue
        if not finite(output.at[parent_index, target]):
            output.at[parent_index, target] = value
        evidence.append(
            {
                "parent_run_id": parent_id,
                "child_run_id": text(child.get("run_id")),
                "child_run_name": text(first(child, "tags.mlflow.runName", "run_name")),
                "source_key": source,
                "recall_at_100": value,
            }
        )
    return output, pd.DataFrame(evidence)


def parent_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if "tags.mlflow.parentRunId" not in runs:
        return runs.copy()
    return runs[runs["tags.mlflow.parentRunId"].isna()].copy()


def is_poster_run(row: pd.Series) -> bool:
    name = text(first(row, "tags.mlflow.runName", "run_name")).lower()
    poster_tag = text(first(row, "tags.poster")).lower()
    return poster_tag in {"true", "1", "yes"} or name.startswith(("poster-", "sid-quality-"))


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_json(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = value
    return flattened


def artifact_sid_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    """Read SID metrics from small JSON artifacts when parent tags are sparse."""
    interesting = {"metrics.json", "result.json", "results.json", "summary.json"}
    queue = [""]
    candidates: list[str] = []
    while queue and len(candidates) < 20:
        path = queue.pop(0)
        try:
            artifacts = client.list_artifacts(run_id, path)
        except Exception:
            break
        for artifact in artifacts:
            if artifact.is_dir and artifact.path.count("/") < 4:
                queue.append(artifact.path)
            elif Path(artifact.path).name.lower() in interesting:
                candidates.append(artifact.path)

    found: dict[str, float] = {}
    for artifact_path in candidates:
        try:
            payload = json.loads(Path(client.download_artifacts(run_id, artifact_path)).read_text(encoding="utf-8"))
        except Exception:
            continue
        values = {
            normalize_key(key): value
            for key, value in flatten_json(payload).items()
            if finite(value)
        }
        for spec in SID_METRICS:
            for key in (spec.requested_key, *spec.aliases):
                normalized_key = normalize_key(key)
                value = values.get(normalized_key)
                if value is None:
                    suffixes = [candidate for flat_key, candidate in values.items() if flat_key.endswith(normalized_key)]
                    value = suffixes[0] if len(suffixes) == 1 else None
                if finite(value):
                    found[spec.column] = float(value)
                    break
    return found


def sid_columns(frame: pl.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if re.fullmatch(r"sid_\d+", column)]
    return sorted(columns, key=lambda column: int(column.removeprefix("sid_")))


def configured_codebook_sizes(config: dict[str, Any], columns: list[str]) -> tuple[int, ...]:
    sizes = config.get("codebook_sizes") or config.get("codebooks_sizes")
    if sizes is not None:
        parsed = tuple(int(size) for size in sizes)
        if len(parsed) == len(columns) and all(size > 0 for size in parsed):
            return parsed
    size = config.get("codebook_size")
    if size is None:
        raise ValueError("SID report has no codebook_size or codebook_sizes")
    parsed_size = int(size)
    if parsed_size <= 0:
        raise ValueError("codebook_size must be positive")
    return (parsed_size,) * len(columns)


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64, copy=False))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    positions = np.arange(1, len(values) + 1, dtype=np.float64)
    return float((2 * np.dot(positions, values) / (len(values) * values.sum())) - (len(values) + 1) / len(values))


def mean_within_group_cosine(
    sids: pl.DataFrame,
    embeddings: pl.DataFrame,
    prefix: list[str],
    embedding_column: str,
) -> float | None:
    grouped = (
        sids.select("item_id", *prefix)
        .join(embeddings.select("item_id", embedding_column), on="item_id", how="inner")
        .group_by(prefix)
        .agg(pl.col(embedding_column).alias("vectors"), pl.len().alias("count"))
        .filter(pl.col("count") > 1)
    )
    similarities = []
    for vectors, count in grouped.select("vectors", "count").iter_rows():
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        # Equivalent to the mean off-diagonal entry of normalized @ normalized.T,
        # without allocating the O(n^2) pairwise matrix for broad prefixes.
        total = normalized.sum(axis=0)
        similarities.append(float((np.dot(total, total) - count) / (count * (count - 1))))
    return float(np.mean(similarities)) if similarities else None


def recompute_sid_metrics(
    semantic_ids_path: Path,
    sid_config: dict[str, Any],
    embeddings: pl.DataFrame | None,
) -> dict[str, float]:
    sids = pl.read_parquet(semantic_ids_path)
    columns = sid_columns(sids)
    if not columns or "item_id" not in sids.columns:
        raise ValueError(f"{semantic_ids_path} has no item_id plus sid_<level> columns")
    sizes = configured_codebook_sizes(sid_config, columns)
    valid = pl.lit(True)
    for column, size in zip(columns, sizes, strict=True):
        valid &= pl.col(column).is_between(0, size - 1, closed="both")
    valid_sids = sids.filter(valid)
    if valid_sids.is_empty():
        raise ValueError(f"{semantic_ids_path} has no valid semantic IDs")

    paths = valid_sids.group_by(columns).len(name="count")
    counts = paths["count"].to_numpy()
    collisions = counts[counts > 1]
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    utilization = [valid_sids[column].n_unique() / size for column, size in zip(columns, sizes, strict=True)]
    metrics: dict[str, float] = {
        "p95_collision_group_size": float(np.quantile(collisions, 0.95, method="nearest")) if len(collisions) else 0.0,
        "cur_total": float(np.mean(utilization)),
        "code_gini": gini(counts),
        "code_entropy": entropy,
        "valid_sid_rate": valid_sids.height / sids.height,
    }
    if embeddings is not None:
        embedding_column = "normalized_embed" if "normalized_embed" in embeddings.columns else "embed"
        for level in range(1, len(columns) + 1):
            coherence = mean_within_group_cosine(valid_sids, embeddings, columns[:level], embedding_column)
            if coherence is not None:
                metrics[f"prefix_coherence_{level}"] = coherence
        if f"prefix_coherence_{len(columns)}" in metrics:
            metrics["intra_code_similarity"] = metrics[f"prefix_coherence_{len(columns)}"]
    return metrics


def local_sid_sources(reports_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    for metrics_path in reports_dir.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            run_id = text(payload.get("tracking", {}).get("run_id"))
            sid_config = payload.get("sid", {}).get("config") or {}
        except Exception:
            continue
        semantic_ids = metrics_path.parent / "semantic_ids.parquet"
        if run_id and semantic_ids.is_file():
            sources[run_id] = (semantic_ids, sid_config)
    return sources


def backfill_local_sid_metrics(
    frame: pd.DataFrame,
    *,
    reports_dir: Path,
    embeddings_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, pd.DataFrame()
    sources = local_sid_sources(reports_dir)
    needs_embeddings = frame.method.isin(["PLUM", "BalanceRQKMeans"]).any()
    if needs_embeddings and not embeddings_path.is_file():
        raise SystemExit(f"Embeddings cache not found: {embeddings_path}")
    embeddings = pl.read_parquet(embeddings_path) if needs_embeddings else None
    output = frame.copy()
    evidence = []
    for index, row in output[output.method.isin(["PLUM", "BalanceRQKMeans"])].iterrows():
        source = sources.get(text(row.run_id))
        if source is None:
            evidence.append({"run_id": text(row.run_id), "status": "missing_local_semantic_ids"})
            continue
        semantic_ids, sid_config = source
        try:
            values = recompute_sid_metrics(semantic_ids, sid_config, embeddings)
        except Exception as error:
            evidence.append({"run_id": text(row.run_id), "status": "calculation_failed", "detail": str(error)})
            continue
        filled = []
        for column, value in values.items():
            if column in output and not finite(output.at[index, column]):
                output.at[index, column] = value
                filled.append(column)
        evidence.append({
            "run_id": text(row.run_id),
            "status": "recomputed",
            "semantic_ids": str(semantic_ids),
            "filled_metrics": ";".join(sorted(filled)),
        })
    return output, pd.DataFrame(evidence)


def config_key(row: pd.Series) -> tuple[Any, ...]:
    sizes = row.get("codebook_sizes") if row.get("sid_family") == "plum" else ""
    return (
        row.get("sid_family"),
        row.get("model"),
        row.get("init"),
        row.get("assignment"),
        row.get("collision_resolution"),
        row.get("num_levels"),
        row.get("codebook_size"),
        sizes,
        row.get("seed"),
        row.get("dataset"),
        row.get("data_scope"),
    )


def records_from_runs(
    runs: pd.DataFrame,
    recall_column: str | None,
    *,
    require_recall: bool = True,
) -> pd.DataFrame:
    records = []
    client = MlflowClient()
    for _, row in parent_runs(runs).iterrows():
        if not is_poster_run(row):
            continue
        recall = row.get(recall_column, np.nan) if recall_column else np.nan
        if require_recall and not finite(recall):
            continue
        config = parse_run_config(row)
        if config["method"] not in METHOD_STYLE:
            continue
        record = {
            **config,
            "run_id": text(row.get("run_id")),
            "status": text(row.get("status")),
            "experiment_name": text(row.get("experiment_name")),
            "start_time": text(row.get("start_time")),
            "recall_at_100": float(recall) if finite(recall) else np.nan,
        }
        for spec in SID_METRICS:
            record[spec.column] = np.nan
        for spec in SID_METRICS:
            column = resolve_metric(row.index, (spec.requested_key, *spec.aliases))
            if column:
                record[spec.column] = row.get(column, np.nan)
        if config["sid_family"] in {"plum", "balance"}:
            missing = [spec for spec in SID_METRICS if not finite(record[spec.column])]
            if missing:
                artifact_values = artifact_sid_metrics(client, record["run_id"])
                for spec in missing:
                    if finite(artifact_values.get(spec.column)):
                        record[spec.column] = artifact_values[spec.column]
                record["artifact_backfilled_metrics"] = ";".join(sorted(artifact_values))
            else:
                record["artifact_backfilled_metrics"] = ""
        else:
            record["artifact_backfilled_metrics"] = ""
        record["_key"] = config_key(pd.Series(record))
        records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    metric_columns = [spec.column for spec in SID_METRICS]
    frame["_metric_count"] = frame[metric_columns].apply(lambda row: sum(finite(value) for value in row), axis=1)
    frame = frame.sort_values(["_metric_count", "start_time"])
    return frame.drop_duplicates("_key", keep="last").drop(columns=["_key", "_metric_count"])


def match_quality(generation: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if generation.empty:
        return generation
    if quality.empty:
        return generation
    quality = quality.copy()
    quality["_key"] = quality.apply(config_key, axis=1)
    lookup = {}
    for _, row in quality.iterrows():
        key = row["_key"]
        if key not in lookup or row["_metric_count"] >= lookup[key]["_metric_count"]:
            lookup[key] = row
    output = []
    for _, row in generation.iterrows():
        item = row.to_dict()
        key = config_key(row)
        quality_row = lookup.get(key)
        if quality_row is not None:
            for spec in SID_METRICS:
                if not finite(item.get(spec.column)) and finite(quality_row.get(spec.column)):
                    item[spec.column] = quality_row[spec.column]
        output.append(item)
    return pd.DataFrame(output)


def padded_limits(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 1.0
    low, high = float(clean.min()), float(clean.max())
    span = high - low
    padding = max(abs(low) * 0.04, 0.01) if span == 0 else span * 0.08
    return low - padding, high + padding


def draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    order_column: str,
    title: str,
    sweep_text: str,
    y_limits: tuple[float, float],
    baseline: float | None,
) -> None:
    for method in METHOD_ORDER:
        points = frame[frame.method.eq(method)].sort_values(order_column)
        points = points.drop_duplicates(order_column).dropna(subset=[x_column, "recall_at_100"])
        if points.empty:
            continue
        color, marker, filled = METHOD_STYLE[method]
        if len(points) > 1:
            ax.plot(points[x_column], points.recall_at_100, color=color, linewidth=1.25, alpha=0.72, zorder=1)
        ax.scatter(
            points[x_column], points.recall_at_100, s=55, marker=marker,
            facecolors=color if filled else "white", edgecolors=color,
            linewidths=1.55, zorder=3,
        )
    if baseline is not None:
        ax.axhline(baseline, color="#7B8794", linestyle=(0, (5, 3)), linewidth=1.2, zorder=0)
    ax.set_title(title, pad=9, fontweight="normal")
    ax.set_ylim(*y_limits)
    ax.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.42)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors="#334155", width=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.text(0.5, -0.18, sweep_text, transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#64748B", clip_on=False)
    ax.text(0.99, 0.985, "better ↗", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#7B8794")


def plot_metric(frame: pd.DataFrame, spec: MetricSpec, output_dir: Path, baseline: float | None) -> bool:
    frame = frame.copy()
    frame[spec.column] = pd.to_numeric(frame[spec.column], errors="coerce")
    frame["recall_at_100"] = pd.to_numeric(frame["recall_at_100"], errors="coerce")
    left = frame[frame.codebook_size.eq(256) & frame.num_levels.isin(LEVEL_SWEEP)].dropna(subset=[spec.column, "recall_at_100"])
    right = frame[frame.num_levels.eq(3) & frame.codebook_size.isin(CODEBOOK_SWEEP)].dropna(subset=[spec.column, "recall_at_100"])
    shown = pd.concat([left, right], ignore_index=True).drop_duplicates("run_id")
    if shown.empty:
        return False

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 13,
        "axes.labelsize": 12, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 14.0, "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, facecolor="white")
    x_limits = padded_limits(shown[spec.column])
    y_values = shown["recall_at_100"]
    if baseline is not None:
        y_values = pd.concat([y_values, pd.Series([baseline])], ignore_index=True)
    y_limits = padded_limits(y_values)
    for axis in axes:
        axis.set_facecolor("white")
        axis.set_xlim(*x_limits)
    draw_panel(axes[0], left, spec.column, "num_levels", "Varying number of codebooks (C=256)", "L: 2 → 3 → 4", y_limits, baseline)
    draw_panel(axes[1], right, spec.column, "codebook_size", "Varying codebook size (L=3)", "C: 128 → 256 → 512 → 1024", y_limits, baseline)
    figure.supxlabel(spec.label, fontsize=12, y=0.145)
    figure.supylabel("Recall@100", fontsize=12, x=0.016)

    present_methods = [method for method in METHOD_ORDER if method in set(shown.method)]
    handles = []
    for method in present_methods:
        color, marker, filled = METHOD_STYLE[method]
        handles.append(Line2D([0], [0], color=color, linewidth=1.1, marker=marker, markersize=8,
                              markerfacecolor=color if filled else "white", markeredgewidth=1.4,
                              label=method))
    if baseline is not None:
        handles.append(Line2D([0], [0], color="#7B8794", linewidth=1.2,
                              linestyle=(0, (5, 3)), label="SASRec baseline"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                  ncol=min(6, max(1, len(handles))), frameon=False, handlelength=1.6,
                  columnspacing=1.0)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.34, wspace=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"recall100_vs_{spec.stem}.{extension}",
                       dpi=300 if extension == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return True


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", frame)] + [(method, frame[frame.method.eq(method)]) for method in METHOD_ORDER]
    for spec in SID_METRICS:
        for group, part in groups:
            pairs = pd.DataFrame({"x": pd.to_numeric(part[spec.column], errors="coerce"),
                                  "y": pd.to_numeric(part["recall_at_100"], errors="coerce")}).dropna()
            pearson = spearman = np.nan
            if len(pairs) >= 2 and pairs.x.nunique() >= 2 and pairs.y.nunique() >= 2:
                pearson = float(pearsonr(pairs.x, pairs.y).statistic)
                spearman = float(spearmanr(pairs.x, pairs.y).statistic)
            rows.append({"sid_metric": spec.column, "sid_metric_label": spec.label,
                         "downstream_metric": "recall_at_100", "group": group,
                         "pearson_r": pearson, "spearman_rho": spearman,
                         "num_points": len(pairs)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--generation-experiments", "--generation-experiment", action="append", default=[])
    parser.add_argument("--quality-experiments", "--quality-experiment", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-recall", type=float, default=0.113)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/poster_runs"),
        help="Local report root containing per-run semantic_ids.parquet files.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path.home() / ".cache/sids_rank_vs_generate/datasets/yambda-50m/embeddings.parquet",
        help="Cached catalog embeddings used for prefix and intra-code coherence.",
    )
    parser.add_argument(
        "--no-local-sid-backfill",
        action="store_true",
        help="Do not recompute missing PLUM/Balance SID metrics from local reports.",
    )
    args = parser.parse_args()

    generation_experiments = split_csv(args.generation_experiments) or ["sid-comparison"]
    quality_experiments = split_csv(args.quality_experiments)
    all_experiments = list(dict.fromkeys([*generation_experiments, *quality_experiments]))
    mlflow.set_tracking_uri(args.tracking_uri)
    runs = load_runs(all_experiments)
    runs, child_evidence = inject_generation_recall(runs)
    metric_columns = [column for column in runs.columns if column.startswith("metrics.")]
    recall_column = resolve_metric(metric_columns, RECALL_KEYS)
    if recall_column is None:
        raise SystemExit("No generation Recall@100 metric found in the requested experiments")

    generation = records_from_runs(runs[runs.experiment_name.isin(generation_experiments)], recall_column)
    if quality_experiments:
        quality = records_from_runs(
            runs[runs.experiment_name.isin(quality_experiments)],
            recall_column,
            require_recall=False,
        )
        if not quality.empty:
            quality["_key"] = quality.apply(config_key, axis=1)
            quality["_metric_count"] = quality[[spec.column for spec in SID_METRICS]].apply(
                lambda row: sum(finite(value) for value in row), axis=1
            )
            generation = match_quality(generation, quality)
    if generation.empty:
        raise SystemExit("No eligible generation runs with SID metrics found")

    local_backfill = pd.DataFrame()
    if not args.no_local_sid_backfill:
        generation, local_backfill = backfill_local_sid_metrics(
            generation,
            reports_dir=args.reports_dir,
            embeddings_path=args.embeddings_path,
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    generation.to_csv(output / "matched_generation_runs.csv", index=False)
    child_evidence.to_csv(output / "generation_child_metric_sources.csv", index=False)
    local_backfill.to_csv(output / "local_sid_metric_backfill.csv", index=False)
    correlations(generation).to_csv(output / "correlations.csv", index=False)

    plotted = []
    # Keep the figure layout compatible with the poster-results directory:
    # each metric is exported directly beside the CSV/metadata artifacts.
    plot_dir = output
    for spec in SID_METRICS:
        if plot_metric(generation, spec, plot_dir, args.baseline_recall):
            plotted.append(spec.column)

    metadata = {
        "tracking_uri": args.tracking_uri,
        "generation_experiments": generation_experiments,
        "quality_experiments": quality_experiments,
        "recall_metric": recall_column,
        "baseline_recall": args.baseline_recall,
        "matched_runs": len(generation),
        "generation_child_metric_rows": len(child_evidence),
        "local_sid_metric_backfill_rows": len(local_backfill),
        "reports_dir": str(args.reports_dir),
        "embeddings_path": str(args.embeddings_path),
        "local_sid_metric_definitions": {
            "p95_collision_group_size": "nearest p95 of full-path buckets with more than one item; 0 when none",
            "cur_total": "mean over SID levels of used_codes / configured_codebook_size",
            "code_gini": "Gini coefficient of full-path item-count distribution",
            "code_entropy": "natural-log Shannon entropy of full-path item-count distribution",
            "prefix_coherence_at_n": "equal-weight mean of within-prefix mean pairwise cosine similarities",
            "intra_code_similarity": "prefix_coherence at the full SID depth",
            "valid_sid_rate": "fraction of rows whose every SID component is within its configured codebook range",
        },
        "plotted_metrics": plotted,
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generation runs: {len(generation)}")
    print(f"Recall@100 metric: {recall_column.removeprefix('metrics.')}")
    print(f"Plots written: {len(plotted)} x PNG/PDF/SVG -> {plot_dir}")
    for method, count in generation.method.value_counts().items():
        print(f"  {method}: {count}")


if __name__ == "__main__":
    main()
