from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from .datasets import DatasetFiles, resolve_dataset
from .data.protocols import (
    POSTER_FULL_33148_PROTOCOL_ID,
    THREE_WAY_PROTOCOL_ID,
    expected_catalog_size,
    validate_protocol_id,
)
from .evaluation.generation_stage import GenerationConfig, GenerationStage
from .evaluation.precomputed_sid_metrics import (
    PrecomputedSidMetricsConfig,
    run_precomputed_sid_metrics,
)
from .evaluation.ranking_stage import RankingConfig, RankingStage
from .evaluation.sid_quality import (
    SID_QUALITY_SCHEMA_VERSION,
    UserCatalogQualityConfig,
)
from .sid_experiment import SidExperimentConfig, run_sid_experiment
from .sids import ISidPipeline, IUserAwareSidPipeline
from .tracking import MLflowConfig, create_tracker, flatten_parameters


@dataclass(frozen=True)
class YambdaDatasetConfig:
    size: str = "50m"
    interactions_path: Path | None = None
    embeddings_path: Path | None = None
    ranking_interactions_path: Path | None = None
    artist_mapping_path: Path | None = None
    album_mapping_path: Path | None = None
    source_embeddings_path: Path | None = None
    core_min_interactions_per_item: int = 5
    history_days: float | None = None
    test_days: float = 7.0
    timestamp_unit_seconds: float = 5.0
    validation_fraction: float = 0.1
    protocol_id: str = THREE_WAY_PROTOCOL_ID

    def __post_init__(self) -> None:
        validate_protocol_id(self.protocol_id)
        if self.protocol_id == POSTER_FULL_33148_PROTOCOL_ID:
            if self.size != "50m":
                raise ValueError("poster protocol requires the 50m dataset")
            if self.core_min_interactions_per_item != 5:
                raise ValueError("poster protocol requires item core 5")
            if self.history_days is not None:
                raise ValueError("poster protocol requires the full history")
            if self.test_days != 7.0:
                raise ValueError("poster protocol requires a 7-day heldout")
            if self.timestamp_unit_seconds != 5.0:
                raise ValueError("poster protocol requires 5-second timestamps")

    def resolve(self) -> DatasetFiles:
        if self.interactions_path is not None or self.embeddings_path is not None:
            if self.interactions_path is None or self.embeddings_path is None:
                raise ValueError(
                    "interactions_path and embeddings_path must be provided together"
                )
            return DatasetFiles(
                size=self.size,
                interactions_path=Path(self.interactions_path),
                embeddings_path=Path(self.embeddings_path),
                ranking_interactions_path=(
                    Path(self.ranking_interactions_path)
                    if self.ranking_interactions_path is not None
                    else None
                ),
                artist_mapping_path=(
                    Path(self.artist_mapping_path)
                    if self.artist_mapping_path is not None
                    else None
                ),
                album_mapping_path=(
                    Path(self.album_mapping_path)
                    if self.album_mapping_path is not None
                    else None
                ),
            )
        return resolve_dataset(
            self.size,
            source_embeddings_path=(
                Path(self.source_embeddings_path)
                if self.source_embeddings_path is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Настройки обучения моделей"""

    ranking: RankingConfig = RankingConfig()
    generation: GenerationConfig = GenerationConfig()


@dataclass(frozen=True)
class ExperimentConfig:
    sid_pipeline: ISidPipeline
    user_sid_pipeline: IUserAwareSidPipeline | None = None
    dataset: YambdaDatasetConfig = YambdaDatasetConfig()
    training: TrainingConfig = TrainingConfig()
    tracking: MLflowConfig = MLflowConfig()
    reports_dir: Path = Path("reports/experiments")
    catalog_from_fit: bool = False
    user_catalog_quality: UserCatalogQualityConfig = UserCatalogQualityConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.sid_pipeline, ISidPipeline):
            raise TypeError("sid_pipeline must implement ISidPipeline")
        if self.user_sid_pipeline is not None and not isinstance(
            self.user_sid_pipeline,
            IUserAwareSidPipeline,
        ):
            raise TypeError("user_sid_pipeline must implement IUserAwareSidPipeline")
        legacy_user_aware = (
            self.dataset.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
            and not self.catalog_from_fit
        )
        if (
            self.user_sid_pipeline is not None
            and not self.catalog_from_fit
            and not legacy_user_aware
        ):
            raise ValueError("user-aware experiments require catalog_from_fit=True")
        if (
            self.dataset.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
            and self.catalog_from_fit
        ):
            raise ValueError("poster protocol requires a full-data catalog")
        if self.user_catalog_quality.enabled and self.user_sid_pipeline is None:
            raise ValueError("user catalog quality requires a user SID pipeline")


def _load_module(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"experiment config not found: {resolved}")
    module_name = f"sids_experiment_{abs(hash(resolved))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import experiment config: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment_config(spec: str) -> ExperimentConfig:
    path_text, separator, attribute = spec.partition(":")
    module = _load_module(Path(path_text))
    attribute = attribute if separator else "EXPERIMENT"
    try:
        config = getattr(module, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"experiment config {path_text!r} has no attribute {attribute!r}"
        ) from error
    if callable(config) and not isinstance(config, ExperimentConfig):
        config = config()
    if not isinstance(config, ExperimentConfig):
        raise TypeError(
            f"{attribute} must be ExperimentConfig, got {type(config).__name__}"
        )
    return config


def run_experiment(
    config: ExperimentConfig,
    *,
    run_ranking: bool = True,
    run_generation: bool = True,
    sid_only: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    if not run_ranking and not run_generation and not sid_only:
        raise ValueError("at least one evaluation stage must run unless sid_only=True")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    dataset = config.dataset.resolve()
    stages = {
        "ranking": RankingStage(config.training.ranking),
        "generation": GenerationStage(config.training.generation),
    }
    enabled_stages = tuple(
        name
        for name, enabled in (
            ("ranking", run_ranking),
            ("generation", run_generation),
        )
        if enabled
    )

    def resolve_stage(name: str):
        return stages[name]

    tracker = create_tracker(config.tracking)
    run_name = config.tracking.run_name or type(config.sid_pipeline).__name__
    tags = {
        "pipeline": type(config.sid_pipeline).__name__,
        "protocol_id": config.dataset.protocol_id,
        "data_scope": (
            "full" if config.dataset.history_days is None else "windowed"
        ),
        "user_pipeline": (
            type(config.user_sid_pipeline).__name__
            if config.user_sid_pipeline is not None
            else "content-only"
        ),
        "quality_metric_schema": SID_QUALITY_SCHEMA_VERSION,
        "user_catalog_quality": (
            "enabled" if config.user_catalog_quality.enabled else "disabled"
        ),
        **dict(config.tracking.tags),
    }
    with tracker.start_run(run_name, tags=tags):
        tracker.log_params(
            flatten_parameters(
                {
                    "dataset": asdict(config.dataset),
                    "data_contract": {
                        "protocol_id": config.dataset.protocol_id,
                        "expected_catalog_size": expected_catalog_size(
                            config.dataset.protocol_id
                        ),
                    },
                    "training": asdict(config.training),
                    "user_sid_pipeline": (
                        type(config.user_sid_pipeline).__name__
                        if config.user_sid_pipeline is not None
                        else None
                    ),
                    "user_catalog_quality": asdict(
                        config.user_catalog_quality
                    ),
                    "enabled_stages": list(enabled_stages),
                    "max_rows": max_rows,
                }
            )
        )
        return run_sid_experiment(
            SidExperimentConfig(
                dataset=dataset,
                sid_pipeline=config.sid_pipeline,
                user_sid_pipeline=config.user_sid_pipeline,
                enabled_stages=enabled_stages,
                reports_dir=config.reports_dir,
                core_min_interactions_per_item=(
                    config.dataset.core_min_interactions_per_item
                ),
                history_days=config.dataset.history_days,
                test_days=config.dataset.test_days,
                timestamp_unit_seconds=config.dataset.timestamp_unit_seconds,
                validation_fraction=config.dataset.validation_fraction,
                catalog_from_fit=config.catalog_from_fit,
                protocol_id=config.dataset.protocol_id,
                user_catalog_quality=config.user_catalog_quality,
                latest_interactions=max_rows,
                tracker=tracker,
            ),
            stage_resolver=resolve_stage,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Python-configured semantic-ID experiment."
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Python config path, optionally followed by :attribute.",
    )
    parser.add_argument(
        "--sids-metrics-dir",
        type=Path,
        help=(
            "Metrics-only mode: scan this directory for precomputed SID "
            "parquets instead of running an experiment config."
        ),
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        help="Embedding parquet used by --sids-metrics-dir.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help=(
            "Output JSON for metrics-only mode; defaults to "
            "<sids-metrics-dir>/sid_metrics.json."
        ),
    )
    parser.add_argument(
        "--recursive-sids",
        action="store_true",
        help="Recursively scan --sids-metrics-dir for parquet files.",
    )
    parser.add_argument(
        "--codebook-sizes",
        help=(
            "Comma-separated codebook sizes for metrics-only mode, such as "
            "256,256,256; defaults to the observed ranges in each SID parquet."
        ),
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Skip both fixed ranking models.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip TIGER-like generation.",
    )
    parser.add_argument(
        "--sid-only",
        action="store_true",
        help="Train the SID and save its item_id -> SID mapping without downstream evaluation.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Use only the latest N rows from each interaction source.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log the complete experiment and every train loop to MLflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking URI; defaults to MLflow's configured local store.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        help="Override the MLflow experiment name from the Python config.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        help="Override the MLflow parent run name from the Python config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sids_metrics_dir is not None:
        if args.config is not None:
            parser.error("config cannot be combined with --sids-metrics-dir")
        if args.embeddings_path is None:
            parser.error("--embeddings-path is required with --sids-metrics-dir")
        incompatible = {
            "--skip-ranking": args.skip_ranking,
            "--skip-generation": args.skip_generation,
            "--max-rows": args.max_rows is not None,
            "--mlflow": args.mlflow,
            "--mlflow-tracking-uri": args.mlflow_tracking_uri is not None,
            "--mlflow-experiment-name": args.mlflow_experiment_name is not None,
            "--mlflow-run-name": args.mlflow_run_name is not None,
        }
        used_incompatible = [
            option for option, enabled in incompatible.items() if enabled
        ]
        if used_incompatible:
            parser.error(
                "experiment options cannot be used in metrics-only mode: "
                + ", ".join(used_incompatible)
            )
        codebook_sizes = None
        if args.codebook_sizes is not None:
            try:
                codebook_sizes = tuple(
                    int(value.strip())
                    for value in args.codebook_sizes.split(",")
                    if value.strip()
                )
            except ValueError:
                parser.error("--codebook-sizes must be comma-separated integers")
            if not codebook_sizes or any(size <= 0 for size in codebook_sizes):
                parser.error("--codebook-sizes must contain positive integers")
        result = run_precomputed_sid_metrics(
            PrecomputedSidMetricsConfig(
                sids_dir=args.sids_metrics_dir,
                embeddings_path=args.embeddings_path,
                output_path=args.metrics_output,
                recursive=args.recursive_sids,
                codebook_sizes=codebook_sizes,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.config is None:
        parser.error("config is required unless --sids-metrics-dir is used")
    metrics_only_options = {
        "--embeddings-path": args.embeddings_path is not None,
        "--metrics-output": args.metrics_output is not None,
        "--recursive-sids": args.recursive_sids,
        "--codebook-sizes": args.codebook_sizes is not None,
    }
    used_metrics_only = [
        option for option, enabled in metrics_only_options.items() if enabled
    ]
    if used_metrics_only:
        parser.error(
            "metrics-only options require --sids-metrics-dir: "
            + ", ".join(used_metrics_only)
        )
    config = load_experiment_config(args.config)
    mlflow_requested = any(
        (
            args.mlflow,
            args.mlflow_tracking_uri is not None,
            args.mlflow_experiment_name is not None,
            args.mlflow_run_name is not None,
        )
    )
    if mlflow_requested:
        config = replace(
            config,
            tracking=replace(
                config.tracking,
                enabled=True,
                tracking_uri=(
                    args.mlflow_tracking_uri
                    if args.mlflow_tracking_uri is not None
                    else config.tracking.tracking_uri
                ),
                experiment_name=(
                    args.mlflow_experiment_name
                    if args.mlflow_experiment_name is not None
                    else config.tracking.experiment_name
                ),
                run_name=(
                    args.mlflow_run_name
                    if args.mlflow_run_name is not None
                    else config.tracking.run_name
                ),
            ),
        )
    result = run_experiment(
        config,
        run_ranking=not args.skip_ranking and not args.sid_only,
        run_generation=not args.skip_generation and not args.sid_only,
        sid_only=args.sid_only,
        max_rows=args.max_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
