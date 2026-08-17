from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl

from .data import PreparedInteractions, load_fit_user_aware_behavior
from .data.protocols import (
    POSTER_FULL_33148_CATALOG_SIZE,
    POSTER_FULL_33148_PROTOCOL_ID,
    THREE_WAY_PROTOCOL_ID,
    validate_poster_full_33148_summary,
    validate_protocol_id,
)
from .datasets import DatasetFiles
from .evaluation.collision_quality import infer_codebook_sizes
from .evaluation.precomputed_sid_metrics import calculate_all_sid_metrics
from .evaluation.sid_quality import (
    SID_QUALITY_SCHEMA_VERSION,
    UserCatalogQualityConfig,
    calculate_user_catalog_sid_quality,
    select_stable_quality_users,
)
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
from .sids.user_aware import (
    IUserAwareSidPipeline,
    UserAwareSidTrainingData,
    user_sid_columns,
)
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
    history_days: float | None = None
    test_days: float = 7.0
    timestamp_unit_seconds: float = 1.0
    validation_fraction: float = 0.1
    catalog_from_fit: bool = False
    latest_interactions: int | None = None
    protocol_id: str = THREE_WAY_PROTOCOL_ID
    user_catalog_quality: UserCatalogQualityConfig = field(
        default_factory=UserCatalogQualityConfig
    )
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
    configuration = getattr(sid, "tracking_config", None)
    if configuration is None:
        configuration = getattr(sid, "config", None)
    if configuration is None:
        return None
    if is_dataclass(configuration) and not isinstance(configuration, type):
        configuration = asdict(cast(Any, configuration))
    elif hasattr(configuration, "__dict__"):
        configuration = vars(configuration)
    return _to_builtin(configuration)


def _user_sid_codebook_sizes(
    user_sid: IUserAwareSidPipeline,
    output: pl.DataFrame,
) -> tuple[int, ...]:
    columns = user_sid_columns(output)
    configuration = getattr(user_sid, "config", None)
    configured = getattr(configuration, "codebook_sizes", None)
    if configured is None:
        size = getattr(configuration, "codebook_size", None)
        levels = getattr(configuration, "num_levels", None)
        if size is not None and levels is not None:
            configured = (int(size),) * int(levels)
    if configured is None:
        raise ValueError(
            "user SID quality requires configured codebook sizes"
        )
    sizes = tuple(int(size) for size in configured)
    if len(sizes) != len(columns):
        raise ValueError(
            "user SID codebook sizes do not match generated SID levels"
        )
    return sizes


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
    if config.user_catalog_quality.enabled and config.user_sid_pipeline is None:
        raise ValueError("user catalog quality requires a user SID pipeline")
    validate_protocol_id(config.protocol_id)
    legacy_user_aware = (
        config.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
        and not config.catalog_from_fit
    )
    if (
        config.user_sid_pipeline is not None
        and not config.catalog_from_fit
        and not legacy_user_aware
    ):
        raise ValueError("user-aware experiments require catalog_from_fit=True")
    if (
        config.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
        and config.latest_interactions is not None
    ):
        raise ValueError("poster protocol does not allow interaction subsampling")

    raw_item_features = _load_item_features(config.dataset.embeddings_path)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=config.dataset.interactions_path,
            core_min_interactions_per_item=config.core_min_interactions_per_item,
            history_days=config.history_days,
            test_days=config.test_days,
            timestamp_unit_seconds=config.timestamp_unit_seconds,
            validation_fraction=config.validation_fraction,
            catalog_from_fit=config.catalog_from_fit,
            latest_interactions=config.latest_interactions,
            protocol_id=config.protocol_id,
        ),
        eligible_items=raw_item_features.select("item_id"),
    )
    if (
        config.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
        and prepared.catalog_size != POSTER_FULL_33148_CATALOG_SIZE
    ):
        raise ValueError(
            "poster protocol catalog mismatch: expected "
            f"{POSTER_FULL_33148_CATALOG_SIZE}, received {prepared.catalog_size}"
        )
    if config.protocol_id == POSTER_FULL_33148_PROTOCOL_ID:
        validate_poster_full_33148_summary(prepared.summary())
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
    configured_sizes = getattr(sid, "sid_codebook_sizes", None)
    codebook_sizes = (
        tuple(configured_sizes)
        if configured_sizes is not None
        else infer_codebook_sizes(getattr(sid, "config", None), sid_output)
    )
    embedding_col = (
        "normalized_embed" if "normalized_embed" in item_features.columns else "embed"
    )
    sid_statistics = calculate_all_sid_metrics(
        sid_output,
        item_features,
        embedding_col=embedding_col,
        codebook_sizes=codebook_sizes,
    )

    user_sid = config.user_sid_pipeline
    fit_user_sids: pl.DataFrame | None = None
    user_sid_statistics: dict[str, Any] | None = None
    user_sid_details: dict[str, Any] | None = None
    user_catalog_quality: dict[str, Any] | None = None
    user_quality_sample: dict[str, Any] | None = None
    behavior_summary: dict[str, Any] | None = None
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
            fit_behavior_interactions = None
            if user_sid.requires_behavior_interactions:
                ranking_path = config.dataset.ranking_interactions_path
                if ranking_path is None:
                    raise ValueError(
                        "user-aware behavior pipeline requires "
                        "ranking_interactions_path"
                    )
                behavior_data = load_fit_user_aware_behavior(
                    ranking_path,
                    prepared,
                    max_rows=config.latest_interactions,
                )
                fit_behavior_interactions = behavior_data.frame
                behavior_summary = behavior_data.summary.as_dict()
            user_sid_tracker.log_params(
                flatten_parameters(
                    {
                        "pipeline": (
                            f"{type(user_sid).__module__}:" f"{type(user_sid).__name__}"
                        ),
                        "config": _sid_configuration(user_sid) or {},
                        "fit_interactions": prepared.fit.height,
                        "catalog_items": item_features.height,
                        "fit_behavior": behavior_summary or {},
                    }
                )
            )
            user_sid.train(
                UserAwareSidTrainingData(
                    fit_interactions=prepared.fit,
                    item_features=item_features,
                    global_sids=sid_output.select("item_id", *sid_columns),
                    fit_end_timestamp=prepared.validation_start_timestamp,
                    fit_behavior_interactions=fit_behavior_interactions,
                )
            )
            fit_pairs = (
                prepared.fit.select("uid", "item_id").unique().sort(["uid", "item_id"])
            )
            fit_user_sids = user_sid.update(fit_pairs)
            user_sid_statistics = calculate_user_sid_statistics(fit_user_sids)
            if config.user_catalog_quality.enabled:
                quality_users, user_quality_sample = select_stable_quality_users(
                    prepared.fit.select("uid"),
                    config.user_catalog_quality,
                )
                user_catalog_quality = calculate_user_catalog_sid_quality(
                    user_sid,
                    quality_users,
                    item_features.select("item_id"),
                    _user_sid_codebook_sizes(user_sid, fit_user_sids),
                    item_features,
                )
                user_catalog_quality["user_sample"] = user_quality_sample
                user_sid_tracker.log_params(
                    flatten_parameters(
                        {
                            "quality": {
                                "schema_version": SID_QUALITY_SCHEMA_VERSION,
                                "config": asdict(config.user_catalog_quality),
                                "sample": user_quality_sample,
                            }
                        }
                    )
                )
            summary = getattr(user_sid, "summary", None)
            if callable(summary):
                user_sid_details = _to_builtin(summary())
            user_sid_tracker.log_metrics(
                flatten_metrics(
                    {
                        "statistics": user_sid_statistics,
                        "catalog_quality": user_catalog_quality or {},
                        "fit_behavior": behavior_summary or {},
                        "details": user_sid_details or {},
                    }
                )
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
    # Keep the canonical internal-catalog artifact used by global SID anchors.
    sid_output.select("item_id", *sid_columns).write_parquet(semantic_ids_path)

    # Export original item IDs separately for downstream mapping analysis.
    sid_mapping_dir = run_dir / "sid_mapping"
    sid_mapping_dir.mkdir(parents=True, exist_ok=True)
    item_sid_mapping_path = sid_mapping_dir / "semantic_ids.parquet"
    (
        prepared.item_mapping.join(
            sid_output.select("item_id", *sid_columns),
            on="item_id",
            how="inner",
        )
        .rename(
            {
                "original_item_id": "item_id",
                "item_id": "catalog_item_idx",
            }
        )
        .select("item_id", "catalog_item_idx", *sid_columns)
        .write_parquet(item_sid_mapping_path)
    )
    user_semantic_ids_path: Path | None = None
    if fit_user_sids is not None:
        user_semantic_ids_path = run_dir / "user_semantic_ids_fit.parquet"
        fit_user_sids.write_parquet(user_semantic_ids_path)
    user_catalog_quality_path: Path | None = None
    if user_catalog_quality is not None:
        user_catalog_quality_path = run_dir / "user_sid_catalog_quality.json"
        user_catalog_quality_path.write_text(
            json.dumps(
                _to_builtin(user_catalog_quality),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
    sid_summary = getattr(sid, "summary", None)
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
                "details": sid_summary() if callable(sid_summary) else None,
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
                    "catalog_quality": user_catalog_quality,
                    "fit_behavior": behavior_summary,
                    "details": user_sid_details,
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
                "protocol_id": config.protocol_id,
                "user_catalog_quality": asdict(config.user_catalog_quality),
                "expected_catalog_size": (
                    POSTER_FULL_33148_CATALOG_SIZE
                    if config.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
                    else None
                ),
            },
            "reproducibility": {"seed": SID_EXPERIMENT_SEED},
            "tracking": config.tracker.info(),
            "artifacts": {
                "run_id": run_id,
                "directory": run_dir,
                "metrics_json": metrics_path,
                "semantic_ids": semantic_ids_path,
                "item_sid_mapping": item_sid_mapping_path,
                "user_semantic_ids_fit": user_semantic_ids_path,
                "user_sid_catalog_quality": user_catalog_quality_path,
            },
        }
    )
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config.tracker.log_artifact(metrics_path)
    if user_catalog_quality_path is not None:
        config.tracker.log_artifact(user_catalog_quality_path)

    # In a complete experiment the parent MLflow run is still active here.
    if config.tracker.info().get("run_id") is not None:
        config.tracker.log_artifact(
            item_sid_mapping_path,
            artifact_path="sid_mapping",
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
                    {
                        "statistics": result["user_sid"]["statistics"],
                        "catalog_quality": result["user_sid"][
                            "catalog_quality"
                        ]
                        or {},
                    }
                    if result["user_sid"] is not None
                    else {}
                ),
                "evaluations": evaluation_metrics,
            }
        )
    )
    return result
