#!/usr/bin/env python3
"""Correlate SID quality with DCNv2/TransAct ranking quality from MLflow.

The analysis is read-only: it reuses completed MLflow runs, never starts
training, and makes missing metrics/configurations explicit in CSV reports.
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
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr


LEVEL_SWEEP = (2, 3, 4)
CODEBOOK_SWEEP = (128, 256, 512, 1024)


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
        "ICR",
        "icr",
        ("analysis.raw_independent_code_rate", "assignment.base_icr"),
    ),
    MetricSpec("sid.collision_rate", "collision_rate", "Collision Rate", "collision_rate"),
    MetricSpec(
        "sid.collision_percentiles.p95",
        "p95_collision_group_size",
        "p95 Collision Group Size",
        "p95_collision_group_size",
    ),
    MetricSpec("sid.max_collision", "g_max", "G_max", "g_max"),
    MetricSpec(
        "sid.codebook_quality.codebook_utilization_rate_total",
        "cur_total",
        "CUR_Total",
        "cur_total",
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

RANKING_METRICS = (
    MetricSpec(
        "evaluations.ranking.dcnv2.test.pairwise_accuracy",
        "dcnv2_pairwise_accuracy",
        "DCNv2 Pairwise Accuracy",
        "dcnv2_pairwise_accuracy",
    ),
    MetricSpec(
        "evaluations.ranking.dcnv2.test.roc_auc",
        "dcnv2_roc_auc",
        "DCNv2 ROC-AUC",
        "dcnv2_roc_auc",
        ("evaluations.ranking.dcnv2.test.auc",),
    ),
    MetricSpec(
        "evaluations.ranking.transact.test.pairwise_accuracy",
        "transact_pairwise_accuracy",
        "TransAct Pairwise Accuracy",
        "transact_pairwise_accuracy",
    ),
    MetricSpec(
        "evaluations.ranking.transact.test.roc_auc",
        "transact_roc_auc",
        "TransAct ROC-AUC",
        "transact_roc_auc",
        ("evaluations.ranking.transact.test.auc",),
    ),
)

# Item-only controls from the matching content-only ranking runs.
RANKING_BASELINES = {
    "dcnv2_pairwise_accuracy": 0.549144446849823,
    "dcnv2_roc_auc": 0.8093270063400269,
    "transact_pairwise_accuracy": 0.7616394758224487,
    "transact_roc_auc": 0.9367052912712097,
}

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

CONFIG_FIELDS = (
    "sid_family",
    "model",
    "init",
    "assignment",
    "collision_resolution",
    "num_levels",
    "codebook_size",
    "codebook_sizes",
    "seed",
    "dataset",
    "data_scope",
    "method",
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


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def split_csv(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(output))


def resolve_metric(columns: Iterable[str], spec: MetricSpec) -> str | None:
    metric_columns = [column for column in columns if column.startswith("metrics.")]
    candidates = (spec.requested_key, *spec.aliases)
    for key in candidates:
        exact = f"metrics.{key}"
        if exact in metric_columns:
            return exact

    normalized = {normalize_key(column.removeprefix("metrics.")): column for column in metric_columns}
    for key in candidates:
        if normalize_key(key) in normalized:
            return normalized[normalize_key(key)]

    # Ranking fallbacks still require model + test + metric tokens, preventing
    # accidental selection of generation Recall or a train metric.
    requested = spec.requested_key.lower()
    if "evaluations.ranking" in requested:
        model = "dcnv2" if "dcnv2" in requested else "transact"
        metric_tokens = ("pairwise", "accuracy") if "pairwise" in requested else ("auc",)
        matches = []
        for column in metric_columns:
            bare = column.lower().removeprefix("metrics.")
            if model not in bare or "ranking" not in bare or "test" not in bare:
                continue
            if all(token in bare for token in metric_tokens):
                matches.append(column)
        if len(matches) == 1:
            return matches[0]
    return None


def parse_run_config(row: pd.Series) -> dict[str, Any]:
    name = text(first(row, "tags.mlflow.runName", "run_name")).lower()

    family = text(first(row, "tags.sid_model", "params.sid_model")).lower()
    if (
        "onerec-pair-rqvae" in name
        or family == "collaborative_rqvae"
    ):
        family = "onerec_pair_rqvae"
    elif (
        "onerec-pair" in name
        or family in {"collaborative_rqkmeans", "onerec_pair"}
    ):
        family = "onerec_pair"
    elif "plum" in name or family == "plum":
        family = "plum"
    elif "balance" in name or family in {"balance", "balance_rqkmeans", "balancerqkmeans"}:
        family = "balance"
    else:
        family = "poster"

    model = text(first(row, "tags.model", "params.config.model", "params.model")).lower()
    init = text(first(row, "tags.init", "params.config.init", "params.init")).lower()
    assignment = text(
        first(row, "tags.assignment", "params.config.assignment", "params.assignment")
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

    original_match = re.search(
        r"(?:poster(?:-ranking)?|sid-quality)-"
        r"(?P<model>rqkmeans|rqvae|r3vae)-"
        r"(?P<init>rqkmeans|spherical)-"
        r"(?P<assignment>nearest|rrs|ecm)"
        r"(?:-(?P<collision>none|token))?",
        name,
    )
    if original_match:
        model = model or original_match.group("model")
        init = init or original_match.group("init")
        assignment = assignment or original_match.group("assignment")
        if original_match.group("collision"):
            collision = original_match.group("collision")

    if family == "onerec_pair_rqvae":
        model, init, assignment = "collaborative_rqvae", "rqkmeans", "nearest"
    elif family == "onerec_pair":
        model, init, assignment = "collaborative_rqkmeans", "rqkmeans", "nearest"
    elif family == "plum":
        model, init, assignment = "plum", "plum", "nearest"
    elif family == "balance":
        model, init, assignment = "balance", "rqkmeans", "nearest"

    level = first(row, "tags.num_levels", "params.config.num_levels", "params.num_levels")
    if level is None:
        match = re.search(r"(?:^|-)l(\d+)(?:-|$)", name)
        level = match.group(1) if match else None

    codebook_size = first(
        row,
        "tags.codebook_size",
        "params.config.codebook_size",
        "params.codebook_size",
    )
    if codebook_size is None:
        match = re.search(r"(?:^|-)c(\d+)(?:eq)?(?:-|$)", name)
        codebook_size = match.group(1) if match else None

    codebook_sizes = text(
        first(row, "tags.codebook_sizes", "params.config.codebook_sizes", "params.codebook_sizes")
    ).lower().replace(",", "x").replace(" ", "")
    if not codebook_sizes:
        match = re.search(r"(?:^|-)cbs([0-9x]+)(?:-|$)", name)
        codebook_sizes = match.group(1) if match else text(codebook_size)

    seed = first(row, "tags.seed", "params.config.seed", "params.seed", default=42)
    dataset = text(
        first(row, "tags.dataset", "params.dataset", "params.dataset.size", "params.config.dataset"),
        "50m",
    ).lower()
    if dataset.isdigit():
        dataset = f"{dataset}m"
    scope = text(first(row, "tags.data_scope", "params.data_scope", "params.config.data_scope"), "full").lower()

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
        method = text(model or family or "unknown")

    return {
        "sid_family": family,
        "model": model,
        "init": init,
        "assignment": assignment,
        "collision_resolution": collision,
        "num_levels": to_int(level),
        "codebook_size": to_int(codebook_size),
        "codebook_sizes": codebook_sizes,
        "seed": to_int(seed, 42),
        "dataset": dataset,
        "data_scope": scope,
        "method": method,
        "run_name": text(first(row, "tags.mlflow.runName", "run_name")),
    }


def load_runs(experiment_names: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    import mlflow

    frames = []
    experiment_ids: dict[str, str] = {}
    for experiment_name in experiment_names:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise SystemExit(f"MLflow experiment not found: {experiment_name}")
        experiment_ids[experiment_name] = experiment.experiment_id
        frame = mlflow.search_runs(
            [experiment.experiment_id], max_results=10000, output_format="pandas"
        )
        frame["experiment_name"] = experiment_name
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True, sort=False)
    runs = runs.drop_duplicates("run_id", keep="last").reset_index(drop=True)
    return runs, experiment_ids


def inject_child_ranking_metrics(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy ranking child-run test metrics onto their parent analysis rows.

    Older poster runs log ``test.pairwise_accuracy`` and ``test.roc_auc`` only
    inside nested ``ranking.dcnv2`` / ``ranking.transact`` runs. Newer runs
    additionally expose flattened ``evaluations.ranking...`` metrics on the
    parent. Supporting both layouts avoids rerunning any model training.
    """
    output = runs.copy()
    parent_column = "tags.mlflow.parentRunId"
    if parent_column not in output:
        return output, pd.DataFrame()

    target_columns = {
        ("dcnv2", "pairwise_accuracy"): "metrics.evaluations.ranking.dcnv2.test.pairwise_accuracy",
        ("dcnv2", "roc_auc"): "metrics.evaluations.ranking.dcnv2.test.roc_auc",
        ("transact", "pairwise_accuracy"): "metrics.evaluations.ranking.transact.test.pairwise_accuracy",
        ("transact", "roc_auc"): "metrics.evaluations.ranking.transact.test.roc_auc",
    }
    for target in target_columns.values():
        if target not in output:
            output[target] = np.nan
    if "ranking_child_run_ids" not in output:
        output["ranking_child_run_ids"] = ""

    parent_indices = {
        text(run_id): index for index, run_id in output["run_id"].items() if text(run_id)
    }
    evidence = []
    for _, child in output[output[parent_column].notna()].iterrows():
        parent_id = text(child.get(parent_column))
        parent_index = parent_indices.get(parent_id)
        if parent_index is None:
            continue
        run_name = text(first(child, "tags.mlflow.runName", "run_name")).lower()
        stage = text(first(child, "tags.stage")).lower()
        model = text(first(child, "tags.model")).lower()
        if not model:
            if "dcnv2" in run_name:
                model = "dcnv2"
            elif "transact" in run_name:
                model = "transact"
        if model not in {"dcnv2", "transact"}:
            continue
        if stage and stage != "ranking" and "ranking" not in run_name:
            continue

        copied = []
        for metric in ("pairwise_accuracy", "roc_auc"):
            source_candidates = [
                f"metrics.test.{metric}",
                f"metrics.{metric}",
                f"metrics.evaluations.ranking.{model}.test.{metric}",
            ]
            if metric == "roc_auc":
                source_candidates.extend(
                    [
                        "metrics.test.auc",
                        "metrics.auc",
                        f"metrics.evaluations.ranking.{model}.test.auc",
                    ]
                )
            value = None
            source = ""
            for candidate in source_candidates:
                if candidate in child and finite(child.get(candidate)):
                    value = float(child[candidate])
                    source = candidate.removeprefix("metrics.")
                    break
            if value is None:
                continue
            target = target_columns[(model, metric)]
            if not finite(output.at[parent_index, target]):
                output.at[parent_index, target] = value
            copied.append(metric)
            evidence.append(
                {
                    "parent_run_id": parent_id,
                    "child_run_id": text(child.get("run_id")),
                    "child_run_name": text(first(child, "tags.mlflow.runName", "run_name")),
                    "child_status": text(child.get("status")),
                    "ranking_model": model,
                    "ranking_metric": metric,
                    "source_key": source,
                    "target_key": target.removeprefix("metrics."),
                    "value": value,
                }
            )
        if copied:
            existing = text(output.at[parent_index, "ranking_child_run_ids"])
            child_id = text(child.get("run_id"))
            ids = [item for item in existing.split(";") if item]
            if child_id not in ids:
                ids.append(child_id)
            output.at[parent_index, "ranking_child_run_ids"] = ";".join(ids)
    return output, pd.DataFrame(evidence)


def eligible_parent_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Select parent runs without mutating MLflow lifecycle state.

    Normally only FINISHED parents are accepted. Some historical poster jobs
    left the outer parent RUNNING even though both FINISHED ranking children
    logged all four test metrics. Those parents are safe for this read-only
    analysis when, and only when, the complete four-metric result is present.
    """
    result = runs.copy()
    if "tags.mlflow.parentRunId" in result:
        result = result[result["tags.mlflow.parentRunId"].isna()]
    if "status" in result:
        required = [f"metrics.{spec.requested_key}" for spec in RANKING_METRICS]
        for column in required:
            if column not in result:
                result[column] = np.nan
        complete_ranking = result[required].apply(
            lambda row: all(finite(value) for value in row), axis=1
        )
        finished = result["status"].eq("FINISHED")
        result = result[finished | complete_ranking].copy()
        result["eligibility_reason"] = np.where(
            result["status"].eq("FINISHED"),
            "parent_finished",
            "complete_ranking_metrics_from_children",
        )
    return result.reset_index(drop=True)


def is_poster_run(row: pd.Series) -> bool:
    name = text(first(row, "tags.mlflow.runName", "run_name")).lower()
    tag = text(first(row, "tags.poster")).lower()
    return tag in {"true", "1", "yes"} or name.startswith(("poster-", "sid-quality-"))


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_json(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value
    return output


def artifact_metric_backfill(client: Any, run_id: str) -> dict[str, float]:
    """Best-effort read of small JSON metric artifacts for PLUM/Balance runs."""
    interesting_names = {"metrics.json", "result.json", "results.json", "summary.json"}
    queue = [""]
    candidates: list[str] = []
    visited = 0
    while queue and visited < 100:
        path = queue.pop(0)
        visited += 1
        try:
            artifacts = client.list_artifacts(run_id, path)
        except Exception:
            break
        for artifact in artifacts:
            if artifact.is_dir and artifact.path.count("/") < 4:
                queue.append(artifact.path)
            elif Path(artifact.path).name.lower() in interesting_names:
                candidates.append(artifact.path)

    found: dict[str, float] = {}
    for artifact_path in candidates[:20]:
        try:
            local_path = client.download_artifacts(run_id, artifact_path)
            payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        flattened = flatten_json(payload)
        normalized = {normalize_key(key): value for key, value in flattened.items() if finite(value)}
        for spec in SID_METRICS:
            for key in (spec.requested_key, *spec.aliases):
                target = normalize_key(key)
                exact = normalized.get(target)
                suffix_matches = [value for flat_key, value in normalized.items() if flat_key.endswith(target)]
                value = exact if exact is not None else (suffix_matches[0] if len(suffix_matches) == 1 else None)
                if finite(value):
                    found[spec.column] = float(value)
                    break
    return found


def collect_candidates(
    runs: pd.DataFrame,
    sid_mapping: dict[str, str | None],
    ranking_mapping: dict[str, str | None],
    *,
    include_token: bool,
    artifact_backfill: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    records: list[dict[str, Any]] = []
    for _, row in eligible_parent_runs(runs).iterrows():
        if not is_poster_run(row):
            continue
        config = parse_run_config(row)
        if config["method"] == "RQVAE-token" and not include_token:
            continue
        if config["method"] not in METHOD_STYLE:
            continue
        record = {
            **config,
            "run_id": text(row.get("run_id")),
            "run_status": text(row.get("status")),
            "eligibility_reason": text(row.get("eligibility_reason"), "parent_finished"),
            "experiment_name": text(row.get("experiment_name")),
            "start_time": text(row.get("start_time")),
            "git_commit": text(row.get("tags.mlflow.source.git.commit")),
            "git_branch": text(row.get("tags.mlflow.source.git.branch")),
            "artifact_uri": text(row.get("artifact_uri")),
            "ranking_child_run_ids": text(row.get("ranking_child_run_ids")),
        }
        for spec in SID_METRICS:
            column = sid_mapping.get(spec.column)
            record[spec.column] = row.get(column, np.nan) if column else np.nan
        for spec in RANKING_METRICS:
            column = ranking_mapping.get(spec.column)
            record[spec.column] = row.get(column, np.nan) if column else np.nan

        has_sid = any(finite(record[spec.column]) for spec in SID_METRICS)
        has_ranking = any(finite(record[spec.column]) for spec in RANKING_METRICS)
        if artifact_backfill and config["sid_family"] in {"plum", "balance"} and has_sid:
            missing = [spec for spec in SID_METRICS if not finite(record[spec.column])]
            if missing:
                backfilled = artifact_metric_backfill(client, record["run_id"])
                for spec in missing:
                    if finite(backfilled.get(spec.column)):
                        record[spec.column] = backfilled[spec.column]
                record["artifact_backfilled_metrics"] = ";".join(sorted(backfilled))
            else:
                record["artifact_backfilled_metrics"] = ""
        else:
            record["artifact_backfilled_metrics"] = ""
        record["has_sid"] = any(finite(record[spec.column]) for spec in SID_METRICS)
        record["has_ranking"] = has_ranking
        if record["has_sid"] or record["has_ranking"]:
            records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame.copy(), frame.copy()
    quality = frame[frame.has_sid].copy()
    ranking = frame[frame.has_ranking].copy()
    return quality, ranking


def match_key(row: pd.Series) -> tuple[Any, ...]:
    # codebook_sizes disambiguates PLUM's effective C from actual per-level sizes.
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


def deduplicate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["_match_key"] = result.apply(match_key, axis=1)
    # Prefer the row with most finite relevant metrics, then the newest run.
    metric_columns = [spec.column for spec in (*SID_METRICS, *RANKING_METRICS)]
    result["_metric_count"] = result[metric_columns].apply(
        lambda row: sum(finite(value) for value in row), axis=1
    )
    result = result.sort_values(["_metric_count", "start_time"], ascending=[True, True])
    return result.drop_duplicates("_match_key", keep="last")


def match_runs(
    quality: pd.DataFrame, ranking: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_unique = deduplicate_candidates(quality)
    ranking_unique = deduplicate_candidates(ranking)
    if quality_unique.empty or ranking_unique.empty:
        return pd.DataFrame(), quality_unique, ranking_unique

    merged = quality_unique.merge(
        ranking_unique,
        on="_match_key",
        how="inner",
        suffixes=("_quality", "_ranking"),
        validate="one_to_one",
    )
    rows = []
    for _, row in merged.iterrows():
        output: dict[str, Any] = {}
        for field in CONFIG_FIELDS:
            output[field] = row.get(f"{field}_quality", row.get(field))
        for spec in SID_METRICS:
            output[spec.column] = row.get(f"{spec.column}_quality", np.nan)
        for spec in RANKING_METRICS:
            output[spec.column] = row.get(f"{spec.column}_ranking", np.nan)
        output.update(
            {
                "sid_quality_run_id": row["run_id_quality"],
                "ranking_run_id": row["run_id_ranking"],
                "sid_quality_run_name": row["run_name_quality"],
                "ranking_run_name": row["run_name_ranking"],
                "quality_experiment": row["experiment_name_quality"],
                "ranking_experiment": row["experiment_name_ranking"],
                "ranking_parent_status": row.get("run_status_ranking", ""),
                "ranking_eligibility_reason": row.get("eligibility_reason_ranking", ""),
                "ranking_child_run_ids": row.get("ranking_child_run_ids_ranking", ""),
                "artifact_backfilled_metrics": row.get("artifact_backfilled_metrics_quality", ""),
                "git_commit_quality": row.get("git_commit_quality", ""),
                "git_branch_quality": row.get("git_branch_quality", ""),
            }
        )
        rows.append(output)
    matched = pd.DataFrame(rows)

    matched_keys = set(merged["_match_key"].tolist())
    unmatched_quality = quality_unique[~quality_unique["_match_key"].isin(matched_keys)].copy()
    unmatched_ranking = ranking_unique[~ranking_unique["_match_key"].isin(matched_keys)].copy()
    for frame in (unmatched_quality, unmatched_ranking):
        frame.drop(columns=["_match_key", "_metric_count"], inplace=True, errors="ignore")
    return matched, unmatched_quality, unmatched_ranking


def numeric_pairs(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def correlation_rows(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [*METHOD_ORDER, "ALL"]
    for ranking_spec in RANKING_METRICS:
        model = "dcnv2" if ranking_spec.column.startswith("dcnv2") else "transact"
        ranking_metric = "pairwise_accuracy" if "pairwise" in ranking_spec.column else "roc_auc"
        for sid_spec in SID_METRICS:
            for group in groups:
                part = matched if group == "ALL" else matched[matched.method.eq(group)]
                pairs = numeric_pairs(part[sid_spec.column], part[ranking_spec.column])
                pearson = np.nan
                spearman = np.nan
                if len(pairs) >= 2 and pairs.x.nunique() >= 2 and pairs.y.nunique() >= 2:
                    pearson = float(pearsonr(pairs.x, pairs.y).statistic)
                    spearman = float(spearmanr(pairs.x, pairs.y).statistic)
                rows.append(
                    {
                        "ranking_model": model,
                        "ranking_metric": ranking_metric,
                        "sid_metric": sid_spec.column,
                        "sid_metric_label": sid_spec.label,
                        "group": group,
                        "pearson_r": pearson,
                        "spearman_rho": spearman,
                        "num_points": len(pairs),
                    }
                )
    return pd.DataFrame(rows)


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
    y_column: str,
    order_column: str,
    title: str,
    sweep_text: str,
    y_limits: tuple[float, float],
    baseline: float | None,
) -> None:
    for method in METHOD_ORDER:
        points = frame[frame.method.eq(method)].sort_values(order_column)
        points = points.drop_duplicates(order_column).dropna(subset=[x_column, y_column])
        if points.empty:
            continue
        color, marker, filled = METHOD_STYLE[method]
        if len(points) > 1:
            ax.plot(
                points[x_column], points[y_column], color=color, linewidth=1.25, alpha=0.72, zorder=1
            )
        ax.scatter(
            points[x_column],
            points[y_column],
            s=55,
            marker=marker,
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidths=1.55,
            zorder=3,
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#7B8794",
            linestyle=(0, (5, 3)),
            linewidth=1.2,
            zorder=0,
        )
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
    ax.text(
        0.5,
        -0.18,
        sweep_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#64748B",
        clip_on=False,
    )


def plot_metric(
    matched: pd.DataFrame,
    sid_spec: MetricSpec,
    ranking_spec: MetricSpec,
    output_dir: Path,
) -> bool:
    columns = [
        sid_spec.column,
        ranking_spec.column,
        "method",
        "num_levels",
        "codebook_size",
    ]
    frame = matched[columns].copy()
    frame[sid_spec.column] = pd.to_numeric(frame[sid_spec.column], errors="coerce")
    frame[ranking_spec.column] = pd.to_numeric(frame[ranking_spec.column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    left = frame[
        frame.codebook_size.eq(256) & frame.num_levels.isin(LEVEL_SWEEP)
    ].dropna(subset=[sid_spec.column, ranking_spec.column])
    right = frame[
        frame.num_levels.eq(3) & frame.codebook_size.isin(CODEBOOK_SWEEP)
    ].dropna(subset=[sid_spec.column, ranking_spec.column])
    shown = pd.concat([left, right], ignore_index=True)
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
            "legend.fontsize": 14.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, facecolor="white")
    baseline = RANKING_BASELINES.get(ranking_spec.column)
    y_values = shown[ranking_spec.column]
    if baseline is not None:
        y_values = pd.concat([y_values, pd.Series([baseline])], ignore_index=True)
    y_limits = padded_limits(y_values)
    x_limits = padded_limits(shown[sid_spec.column])
    for axis in axes:
        axis.set_facecolor("white")
        axis.set_xlim(*x_limits)
    draw_panel(
        axes[0],
        left,
        sid_spec.column,
        ranking_spec.column,
        "num_levels",
        "Varying number of codebooks (C=256)",
        "L: 2 → 3 → 4",
        y_limits,
        baseline,
    )
    draw_panel(
        axes[1],
        right,
        sid_spec.column,
        ranking_spec.column,
        "codebook_size",
        "Varying codebook size (L=3)",
        "C: 128 → 256 → 512 → 1024",
        y_limits,
        baseline,
    )
    figure.supxlabel(sid_spec.label, fontsize=12, y=0.145)
    figure.supylabel(ranking_spec.label, fontsize=12, x=0.016)

    present_methods = [method for method in METHOD_ORDER if method in set(shown.method)]
    handles = []
    for method in present_methods:
        color, marker, filled = METHOD_STYLE[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.1,
                marker=marker,
                markersize=8,
                markerfacecolor=color if filled else "white",
                markeredgewidth=1.4,
                label=method,
            )
        )
    if baseline is not None:
        baseline_label = (
            "DCNv2 item-only"
            if ranking_spec.column.startswith("dcnv2")
            else "TransAct item-only"
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="#7B8794",
                linewidth=1.2,
                linestyle=(0, (5, 3)),
                label=baseline_label,
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(6, max(1, len(handles))),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.34, wspace=0.06)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ranking_spec.stem}_vs_{sid_spec.stem}"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return True


def missing_combinations(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        if not matched.method.eq(method).any():
            continue
        for level in LEVEL_SWEEP:
            if not (
                matched.method.eq(method)
                & matched.num_levels.eq(level)
                & matched.codebook_size.eq(256)
            ).any():
                rows.append(
                    {
                        "method": method,
                        "panel": "varying_levels",
                        "num_levels": level,
                        "codebook_size": 256,
                        "reason": "missing configuration",
                    }
                )
        for codebook_size in CODEBOOK_SWEEP:
            if not (
                matched.method.eq(method)
                & matched.num_levels.eq(3)
                & matched.codebook_size.eq(codebook_size)
            ).any():
                rows.append(
                    {
                        "method": method,
                        "panel": "varying_codebook_size",
                        "num_levels": 3,
                        "codebook_size": codebook_size,
                        "reason": "missing configuration",
                    }
                )
    return pd.DataFrame(rows)


def metric_coverage(
    matched: pd.DataFrame,
    sid_mapping: dict[str, str | None],
    ranking_mapping: dict[str, str | None],
) -> pd.DataFrame:
    rows = []
    for source, specs, mapping in (
        ("sid_quality", SID_METRICS, sid_mapping),
        ("ranking", RANKING_METRICS, ranking_mapping),
    ):
        for spec in specs:
            values = pd.to_numeric(matched.get(spec.column, pd.Series(dtype=float)), errors="coerce")
            overall_finite = int(values.replace([np.inf, -np.inf], np.nan).notna().sum())
            resolved = mapping.get(spec.column)
            rows.append(
                {
                    "source": source,
                    "metric": spec.column,
                    "requested_key": spec.requested_key,
                    "resolved_key": resolved.removeprefix("metrics.") if resolved else "",
                    "method": "ALL",
                    "matched_runs": len(matched),
                    "finite_values": overall_finite,
                    "missing_values": len(matched) - overall_finite,
                    "status": "resolved" if resolved else "missing",
                }
            )
            for method in METHOD_ORDER:
                part = matched[matched.method.eq(method)]
                if part.empty:
                    continue
                part_values = pd.to_numeric(part[spec.column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                rows.append(
                    {
                        "source": source,
                        "metric": spec.column,
                        "requested_key": spec.requested_key,
                        "resolved_key": resolved.removeprefix("metrics.") if resolved else "",
                        "method": method,
                        "matched_runs": len(part),
                        "finite_values": int(part_values.notna().sum()),
                        "missing_values": int(part_values.isna().sum()),
                        "status": "resolved" if resolved else "missing",
                    }
                )
    return pd.DataFrame(rows)


def git_value(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument(
        "--quality-experiments",
        action="append",
        default=[],
        help="Comma-separated MLflow experiment names; can be repeated.",
    )
    parser.add_argument(
        "--ranking-experiments",
        action="append",
        default=[],
        help="Comma-separated MLflow experiment names; can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-token",
        action="store_true",
        help="Include generation-specific RQVAE-token runs if ranking metrics exist.",
    )
    parser.add_argument(
        "--no-artifact-backfill",
        action="store_true",
        help="Do not inspect small JSON artifacts for missing PLUM/Balance SID metrics.",
    )
    args = parser.parse_args()

    quality_experiments = split_csv(args.quality_experiments) or ["sid-quality-only-full-v2", "test"]
    ranking_experiments = split_csv(args.ranking_experiments) or ["sid-comparison", "test"]
    experiment_names = list(dict.fromkeys([*quality_experiments, *ranking_experiments]))

    mlflow.set_tracking_uri(args.tracking_uri)
    all_runs, experiment_ids = load_runs(experiment_names)
    all_runs, child_ranking_evidence = inject_child_ranking_metrics(all_runs)
    metric_columns = [column for column in all_runs.columns if column.startswith("metrics.")]
    sid_mapping = {spec.column: resolve_metric(metric_columns, spec) for spec in SID_METRICS}
    ranking_mapping = {spec.column: resolve_metric(metric_columns, spec) for spec in RANKING_METRICS}

    print("Resolved MLflow metrics:")
    print(
        "  DCNv2 pairwise_accuracy ->",
        (ranking_mapping["dcnv2_pairwise_accuracy"] or "<missing>").removeprefix("metrics."),
    )
    print(
        "  DCNv2 roc_auc           ->",
        (ranking_mapping["dcnv2_roc_auc"] or "<missing>").removeprefix("metrics."),
    )
    print(
        "  TransAct pairwise_accuracy ->",
        (ranking_mapping["transact_pairwise_accuracy"] or "<missing>").removeprefix("metrics."),
    )
    print(
        "  TransAct roc_auc           ->",
        (ranking_mapping["transact_roc_auc"] or "<missing>").removeprefix("metrics."),
    )
    print("Resolved SID metrics:")
    for spec in SID_METRICS:
        resolved = sid_mapping[spec.column]
        print(f"  {spec.label:<28} -> {(resolved or '<missing>').removeprefix('metrics.')}")

    quality, ranking = collect_candidates(
        all_runs,
        sid_mapping,
        ranking_mapping,
        include_token=args.include_token,
        artifact_backfill=not args.no_artifact_backfill,
    )
    matched, unmatched_quality, unmatched_ranking = match_runs(quality, ranking)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / "matched_runs.csv", index=False)
    unmatched_quality.to_csv(output_dir / "unmatched_sid_quality_runs.csv", index=False)
    unmatched_ranking.to_csv(output_dir / "unmatched_ranking_runs.csv", index=False)
    child_ranking_evidence.to_csv(output_dir / "ranking_child_metric_sources.csv", index=False)
    missing_combinations(matched).to_csv(output_dir / "missing_combinations.csv", index=False)
    coverage = metric_coverage(matched, sid_mapping, ranking_mapping)
    coverage.to_csv(output_dir / "metric_coverage_summary.csv", index=False)
    correlations = correlation_rows(matched)
    correlations.to_csv(output_dir / "correlations.csv", index=False)

    plotted = []
    for ranking_spec in RANKING_METRICS:
        model = "dcnv2" if ranking_spec.column.startswith("dcnv2") else "transact"
        metric = "pairwise_accuracy" if "pairwise" in ranking_spec.column else "roc_auc"
        for sid_spec in SID_METRICS:
            if plot_metric(matched, sid_spec, ranking_spec, output_dir / model / metric):
                plotted.append(f"{ranking_spec.column}:{sid_spec.column}")

    method_counts = matched.method.value_counts().to_dict() if not matched.empty else {}
    metadata = {
        "tracking_uri": args.tracking_uri,
        "experiments": experiment_ids,
        "quality_experiments": quality_experiments,
        "ranking_experiments": ranking_experiments,
        "metric_mappings": {
            "sid": sid_mapping,
            "ranking": ranking_mapping,
        },
        "ranking_baselines": RANKING_BASELINES,
        "filters": {
            "parent_only": True,
            "status": "FINISHED",
            "ranking_metric_required": True,
            "include_token": args.include_token,
            "artifact_backfill": not args.no_artifact_backfill,
        },
        "candidate_quality_runs": len(quality),
        "candidate_ranking_runs": len(ranking),
        "ranking_child_metric_rows": len(child_ranking_evidence),
        "ranking_parents_enriched_from_children": int(
            child_ranking_evidence["parent_run_id"].nunique()
        )
        if not child_ranking_evidence.empty
        else 0,
        "matched_runs": len(matched),
        "unmatched_sid_quality_runs": len(unmatched_quality),
        "unmatched_ranking_runs": len(unmatched_ranking),
        "method_counts": method_counts,
        "selected_runs": matched[
            ["method", "num_levels", "codebook_size", "sid_quality_run_id", "ranking_run_id"]
        ].to_dict(orient="records")
        if not matched.empty
        else [],
        "plotted_combinations": plotted,
        "expected_output_figures_per_format": len(plotted),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "git_branch": git_value(Path(__file__).resolve().parents[1], "branch", "--show-current"),
        "logq": "not_requested_for_ranking",
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nData integrity summary:")
    print(f"  SID quality candidate runs: {len(quality)}")
    print(f"  Ranking candidate runs:     {len(ranking)}")
    print(
        "  Ranking parents from child runs:",
        child_ranking_evidence["parent_run_id"].nunique()
        if not child_ranking_evidence.empty
        else 0,
    )
    print(f"  Matched runs:               {len(matched)}")
    print(f"  Unmatched SID quality:      {len(unmatched_quality)}")
    print(f"  Unmatched ranking:          {len(unmatched_ranking)}")
    print("  Matched by method:")
    for method in METHOD_ORDER:
        if method_counts.get(method):
            print(f"    {method:<22} {method_counts[method]}")
    print("\nMetric coverage (ALL):")
    print(
        coverage[coverage.method.eq("ALL")][
            ["source", "metric", "resolved_key", "finite_values", "missing_values", "status"]
        ].to_string(index=False)
    )
    print(f"\nPlots written per format: {len(plotted)}")


if __name__ == "__main__":
    main()
