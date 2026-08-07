from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl

from .data import PreparedInteractions
from .datasets import DatasetFiles
from .evaluation.collision_quality import (
    codebook_quality,
    infer_codebook_sizes,
    path_average_similarity,
)
from .evaluation.sid_statistics import calculate_sid_statistics
from .evaluation.user_sid_statistics import calculate_user_sid_statistics
from .evaluation.stages import (
    EvaluationContext,
    IEvaluationStage,
    create_evaluation_stage,
)
from .pipeline import PipelineConfig, run_pipeline
from .plugins import load_plugin
from .reproducibility import SID_EXPERIMENT_SEED, seed_everything
from .sids.base import ISidPipeline
from .sids.user_aware import IUserAwareSidPipeline, UserAwareSidTrainingData
from .tracking import (
    ExperimentTracker,
    NullTracker,
    flatten_metrics,
    flatten_parameters,
)


@dataclass(frozen=True)
class SidExperimentConfig:
    dataset: DatasetFiles
    sid_spec: str | None = None
    sid_pipeline: ISidPipeline | None = None
    user_sid_pipeline: IUserAwareSidPipeline | None = None
    enabled_stages: tuple[str, ...] = ()
    reports_dir: Path = Path("reports/experiments")
    core_min_interactions_per_item: int = 5
    test_days: float = 7.0
    timestamp_unit_seconds: float = 1.0
    validation_fraction: float = 0.1
    catalog_from_fit: bool = False
    latest_interactions: int | None = None
    tracker: ExperimentTracker = field(default_factory=NullTracker)


def _load_item_features(path: Path) -> pl.DataFrame:
    embeddings_path = path.expanduser()
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"item embeddings parquet not found: {embeddings_path}")

    features = pl.read_parquet(embeddings_path)
    if features.is_empty():
        raise ValueError("item embeddings must not be empty")
    if "item_id" not in features.columns:
        raise ValueError("item embeddings must contain an item_id column")
    if features["item_id"].null_count():
        raise ValueError("item embeddings contain null item_id values")
    if features["item_id"].n_unique() != features.height:
        raise ValueError("item embeddings must contain exactly one row per item_id")

    return features


def _map_item_features(
    features: pl.DataFrame,
    prepared: PreparedInteractions,
) -> pl.DataFrame:
    feature_columns = [column for column in features.columns if column != "item_id"]
    mapping = prepared.item_mapping.rename({"item_id": "mapped_item_id"})
    mapped = (
        features.rename({"item_id": "original_item_id"})
        .join(mapping, on="original_item_id", how="inner")
        .select("mapped_item_id", *feature_columns)
        .rename({"mapped_item_id": "item_id"})
        .sort("item_id")
    )
    if mapped.height != prepared.catalog_size:
        missing = prepared.catalog_size - mapped.height
        raise ValueError(
            f"item embeddings are missing {missing} items from the prepared catalog"
        )
    return mapped


def _sid_columns(frame: pl.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("sid_") and column[4:].isdigit()
    ]
    return sorted(columns, key=lambda column: int(column[4:]))


def _validate_sid_output(
    source_items: pl.DataFrame,
    updated_items: pl.DataFrame,
) -> list[str]:
    if not isinstance(updated_items, pl.DataFrame):
        raise TypeError("SID update() must return a polars DataFrame")
    if "item_id" not in updated_items.columns:
        raise ValueError("SID output must contain item_id")
    if updated_items.height != source_items.height:
        raise ValueError("SID output must contain exactly one row per input item")
    if updated_items["item_id"].null_count():
        raise ValueError("SID output contains null item_id values")
    if updated_items["item_id"].n_unique() != updated_items.height:
        raise ValueError("SID output contains duplicate item_id values")
    if set(updated_items["item_id"].to_list()) != set(
        source_items["item_id"].to_list()
    ):
        raise ValueError("SID output item_id values do not match the prepared catalog")

    columns = _sid_columns(updated_items)
    if not columns:
        raise ValueError("SID output must add at least one sid_<level> column")
    for column in columns:
        if updated_items[column].null_count():
            raise ValueError(f"SID output column {column} contains nulls")
        if not updated_items.schema[column].is_integer():
            raise TypeError(f"SID output column {column} must contain integer codes")
    return columns


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_builtin(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("experiment results must not contain non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value_type = type(value).__name__
    raise TypeError(f"experiment result is not JSON serializable: {value_type}")


def _sid_configuration(sid: Any) -> Any:
    configuration = getattr(sid, "config", None)
    if configuration is None:
        return None
    if is_dataclass(configuration) and not isinstance(configuration, type):
        configuration = asdict(cast(Any, configuration))
    elif hasattr(configuration, "__dict__"):
        configuration = vars(configuration)
    return _to_builtin(configuration)


def run_sid_experiment(
    config: SidExperimentConfig,
    *,
    stage_resolver: Callable[[str], IEvaluationStage] = create_evaluation_stage,
) -> dict[str, Any]:
    unknown_stages = set(config.enabled_stages) - {
        "ranking",
        "generation",
        "candgen",
    }
    if unknown_stages:
        raise ValueError(
            f"unknown evaluation stages: {', '.join(sorted(unknown_stages))}"
        )
    if len(set(config.enabled_stages)) != len(config.enabled_stages):
        raise ValueError("evaluation stages must not be repeated")
    if config.user_sid_pipeline is not None and not config.catalog_from_fit:
        raise ValueError("user-aware experiments require catalog_from_fit=True")

    raw_item_features = _load_item_features(config.dataset.embeddings_path)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=config.dataset.interactions_path,
            core_min_interactions_per_item=config.core_min_interactions_per_item,
            test_days=config.test_days,
            timestamp_unit_seconds=config.timestamp_unit_seconds,
            validation_fraction=config.validation_fraction,
            catalog_from_fit=config.catalog_from_fit,
            latest_interactions=config.latest_interactions,
        ),
        eligible_items=raw_item_features.select("item_id"),
    )
    item_features = _map_item_features(raw_item_features, prepared)
    if config.sid_pipeline is not None and config.sid_spec is not None:
        raise ValueError("provide sid_pipeline or sid_spec, not both")
    if config.sid_pipeline is not None:
        sid = config.sid_pipeline
        sid_reference = f"{type(sid).__module__}:{type(sid).__name__}"
    elif config.sid_spec is not None:
        sid = load_plugin(config.sid_spec, cast(Any, ISidPipeline))
        sid_reference = config.sid_spec
    else:
        raise ValueError("sid_pipeline is required")
    seed_everything(SID_EXPERIMENT_SEED)
    with config.tracker.start_run(
        "sid",
        nested=True,
        tags={"stage": "sid", "pipeline": type(sid).__name__},
    ) as sid_tracker:
        sid.set_tracker(sid_tracker)
        sid_tracker.log_params(
            flatten_parameters(
                {
                    "pipeline": sid_reference,
                    "config": _sid_configuration(sid) or {},
                    "catalog_items": item_features.height,
                }
            )
        )
        sid.train(item_features)
        sid_output = sid.update(item_features)
    sid_columns = _validate_sid_output(item_features, sid_output)
    items_with_sids = item_features.join(
        sid_output.select("item_id", *sid_columns),
        on="item_id",
        how="inner",
    )
    sid_statistics = calculate_sid_statistics(
        sid_output.select("item_id", *sid_columns)
    )
    codebook_sizes = infer_codebook_sizes(getattr(sid, "config", None), sid_output)
    sid_statistics["path_average_similarity"] = path_average_similarity(
        sid_output,
        item_features,
        embedding_col=(
            "normalized_embed"
            if "normalized_embed" in item_features.columns
            else "embed"
        ),
    )
    sid_statistics["codebook_quality"] = codebook_quality(
        sid_output,
        codebook_sizes,
    )

    user_sid = config.user_sid_pipeline
    fit_user_sids: pl.DataFrame | None = None
    user_sid_statistics: dict[str, Any] | None = None
    if user_sid is not None:
        seed_everything(SID_EXPERIMENT_SEED)
        with config.tracker.start_run(
            "user_sid",
            nested=True,
            tags={
                "stage": "user_sid",
                "pipeline": type(user_sid).__name__,
            },
        ) as user_sid_tracker:
            user_sid.set_tracker(user_sid_tracker)
            user_sid_tracker.log_params(
                flatten_parameters(
                    {
                        "pipeline": (
                            f"{type(user_sid).__module__}:" f"{type(user_sid).__name__}"
                        ),
                        "config": _sid_configuration(user_sid) or {},
                        "fit_interactions": prepared.fit.height,
                        "catalog_items": item_features.height,
                    }
                )
            )
            user_sid.train(
                UserAwareSidTrainingData(
                    fit_interactions=prepared.fit,
                    item_features=item_features,
                    global_sids=sid_output.select("item_id", *sid_columns),
                    fit_end_timestamp=prepared.validation_start_timestamp,
                )
            )
            fit_pairs = (
                prepared.fit.select("uid", "item_id").unique().sort(["uid", "item_id"])
            )
            fit_user_sids = user_sid.update(fit_pairs)
            user_sid_statistics = calculate_user_sid_statistics(fit_user_sids)
            user_sid_tracker.log_metrics(
                flatten_metrics({"statistics": user_sid_statistics})
            )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        config.reports_dir.expanduser()
        / config.dataset.size
        / type(sid).__name__
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    semantic_ids_path = run_dir / "semantic_ids.parquet"
    sid_output.select("item_id", *sid_columns).write_parquet(semantic_ids_path)
    user_semantic_ids_path: Path | None = None
    if fit_user_sids is not None:
        user_semantic_ids_path = run_dir / "user_semantic_ids_fit.parquet"
        fit_user_sids.write_parquet(user_semantic_ids_path)
    evaluations: dict[str, Any] = {}
    for stage_name in config.enabled_stages:
        evaluator = stage_resolver(stage_name)
        if evaluator.name != stage_name:
            raise ValueError(
                f"stage resolver returned {evaluator.name!r}; expected {stage_name!r}"
            )
        stage_artifact_dir = run_dir / stage_name
        stage_artifact_dir.mkdir(parents=True, exist_ok=True)
        context = EvaluationContext(
            prepared=prepared,
            items_with_sids=items_with_sids,
            dataset=config.dataset,
            artifact_dir=stage_artifact_dir,
            max_rows=config.latest_interactions,
            tracker=config.tracker,
            user_sid_pipeline=user_sid,
            fit_user_sids=fit_user_sids,
        )
        metrics = evaluator.evaluate(context)
        if not isinstance(metrics, Mapping):
            raise TypeError(f"evaluation stage {stage_name} must return a mapping")
        evaluations[stage_name] = _to_builtin(metrics)

    metrics_path = run_dir / "metrics.json"
    result = _to_builtin(
        {
            "experiment": "sid_evaluation",
            "data": prepared.summary(),
            "sid": {
                "pipeline": sid_reference,
                "class": type(sid).__name__,
                "config": _sid_configuration(sid),
                "item_count": items_with_sids.height,
                "columns": sid_columns,
                "statistics": sid_statistics,
            },
            "user_sid": (
                None
                if user_sid is None
                else {
                    "pipeline": (
                        f"{type(user_sid).__module__}:{type(user_sid).__name__}"
                    ),
                    "class": type(user_sid).__name__,
                    "config": _sid_configuration(user_sid),
                    "statistics": user_sid_statistics,
                }
            ),
            "item_features": {
                "source_items": raw_item_features.height,
                "catalog_items": item_features.height,
                "columns": [
                    column for column in item_features.columns if column != "item_id"
                ],
            },
            "evaluations": evaluations,
            "config": {
                "dataset_size": config.dataset.size,
                "core_min_interactions_per_item": (
                    config.core_min_interactions_per_item
                ),
                "test_days": config.test_days,
                "timestamp_unit_seconds": config.timestamp_unit_seconds,
                "validation_fraction": config.validation_fraction,
                "catalog_from_fit": config.catalog_from_fit,
                "latest_interactions": config.latest_interactions,
                "enabled_stages": list(config.enabled_stages),
            },
            "reproducibility": {"seed": SID_EXPERIMENT_SEED},
            "tracking": config.tracker.info(),
            "artifacts": {
                "run_id": run_id,
                "directory": run_dir,
                "metrics_json": metrics_path,
                "semantic_ids": semantic_ids_path,
                "user_semantic_ids_fit": user_semantic_ids_path,
            },
        }
    )
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evaluation_metrics: dict[str, Any] = {}
    for stage_name, stage_result in evaluations.items():
        if isinstance(stage_result, Mapping) and "metrics" in stage_result:
            evaluation_metrics[stage_name] = stage_result["metrics"]
        elif isinstance(stage_result, Mapping) and "models" in stage_result:
            evaluation_metrics[stage_name] = stage_result["models"]
        else:
            evaluation_metrics[stage_name] = stage_result
    config.tracker.log_metrics(
        flatten_metrics(
            {
                "data": result["data"],
                "sid": result["sid"]["statistics"],
                "user_sid": (
                    result["user_sid"]["statistics"]
                    if result["user_sid"] is not None
                    else {}
                ),
                "evaluations": evaluation_metrics,
            }
        )
    )
    return result
