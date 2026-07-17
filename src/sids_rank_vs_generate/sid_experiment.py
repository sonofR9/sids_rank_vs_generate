from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .data import PreparedInteractions
from .evaluation.stages import EvaluationContext, IEvaluationStage
from .pipeline import PipelineConfig, run_pipeline
from .plugins import load_plugin
from .sids.base import ISidPipeline


@dataclass(frozen=True)
class SidExperimentConfig:
    interactions_path: Path
    embeddings_path: Path
    sid_spec: str
    evaluator_specs: Mapping[str, str]
    core_min_interactions_per_item: int = 5
    test_days: float = 7.0
    seed: int = 42


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
    raise TypeError(f"experiment result is not JSON serializable: {type(value).__name__}")


def run_sid_experiment(config: SidExperimentConfig) -> dict[str, Any]:
    raw_item_features = _load_item_features(config.embeddings_path)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=config.interactions_path,
            core_min_interactions_per_item=config.core_min_interactions_per_item,
            test_days=config.test_days,
            seed=config.seed,
        ),
        eligible_items=raw_item_features.select("item_id"),
    )
    item_features = _map_item_features(raw_item_features, prepared)
    sid = load_plugin(config.sid_spec, ISidPipeline)
    sid.train(item_features)
    sid_output = sid.update(item_features)
    sid_columns = _validate_sid_output(item_features, sid_output)
    items_with_sids = item_features.join(
        sid_output.select("item_id", *sid_columns),
        on="item_id",
        how="inner",
    )

    context = EvaluationContext(
        prepared=prepared,
        items_with_sids=items_with_sids,
        seed=config.seed,
    )
    evaluations: dict[str, Any] = {}
    for requested_name, evaluator_spec in config.evaluator_specs.items():
        evaluator = load_plugin(evaluator_spec, IEvaluationStage)
        if evaluator.name != requested_name:
            raise ValueError(
                f"evaluator {evaluator_spec!r} declares name {evaluator.name!r}; "
                f"expected {requested_name!r}"
            )
        metrics = evaluator.evaluate(context)
        if not isinstance(metrics, Mapping):
            raise TypeError(f"evaluation stage {requested_name} must return a mapping")
        evaluations[requested_name] = _to_builtin(metrics)

    return _to_builtin(
        {
            "experiment": "sid_evaluation",
            "data": prepared.summary(),
            "sid": {
                "plugin": config.sid_spec,
                "class": type(sid).__name__,
                "item_count": items_with_sids.height,
                "columns": sid_columns,
            },
            "item_features": {
                "source_items": raw_item_features.height,
                "catalog_items": item_features.height,
                "columns": [
                    column
                    for column in item_features.columns
                    if column != "item_id"
                ],
            },
            "evaluations": evaluations,
            "config": {
                "interactions_path": config.interactions_path,
                "embeddings_path": config.embeddings_path,
                "core_min_interactions_per_item": (
                    config.core_min_interactions_per_item
                ),
                "test_days": config.test_days,
                "seed": config.seed,
                "evaluators": dict(config.evaluator_specs),
            },
        }
    )


def _flatten_metrics(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_metrics(next_prefix, item))
        return rows
    return [(prefix, value)]


def render_sid_report(result: Mapping[str, Any]) -> str:
    sid = result["sid"]
    data = result["data"]
    lines = [
        f"# SID experiment: {sid['class']}",
        "",
        "## Summary",
        "",
        f"- SID plugin: `{sid['plugin']}`",
        f"- Items with semantic IDs: {sid['item_count']}",
        f"- SID columns: {', '.join(sid['columns'])}",
        f"- Seed: {result['config']['seed']}",
        "",
        "## Data",
        "",
        "| Field | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in data.items())
    item_features = result.get("item_features", {})
    if item_features:
        lines.extend(
            [
                "",
                "## Item features",
                "",
                f"- Source items: {item_features['source_items']}",
                f"- Items in the prepared catalog: {item_features['catalog_items']}",
                f"- Feature columns: {', '.join(item_features['columns'])}",
            ]
        )
    lines.extend(["", "## Evaluation", ""])

    evaluations = result["evaluations"]
    if not evaluations:
        lines.append("No ranking or candidate-generation evaluator was selected.")
    for stage_name, metrics in evaluations.items():
        lines.extend(
            [
                f"### {stage_name}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        lines.extend(
            f"| {name} | {value} |"
            for name, value in _flatten_metrics("", metrics)
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_sid_artifacts(
    result: Mapping[str, Any],
    metrics_json: Path | None = None,
    report_md: Path | None = None,
) -> None:
    if metrics_json is not None:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if report_md is not None:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(render_sid_report(result), encoding="utf-8")
