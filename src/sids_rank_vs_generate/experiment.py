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
from .evaluation.generation_stage import GenerationConfig, GenerationStage
from .evaluation.ranking_stage import RankingConfig, RankingStage
from .sid_experiment import SidExperimentConfig, run_sid_experiment
from .sids import ISidPipeline
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
    history_days: float | None = 365.0
    test_days: float = 7.0
    timestamp_unit_seconds: float = 5.0

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
    """Optional hyperparameter overrides; model and stage choices stay fixed."""

    ranking: RankingConfig = RankingConfig()
    generation: GenerationConfig = GenerationConfig()


@dataclass(frozen=True)
class ExperimentConfig:
    sid_pipeline: ISidPipeline
    dataset: YambdaDatasetConfig = YambdaDatasetConfig()
    training: TrainingConfig = TrainingConfig()
    tracking: MLflowConfig = MLflowConfig()
    reports_dir: Path = Path("reports/experiments")

    def __post_init__(self) -> None:
        if not isinstance(self.sid_pipeline, ISidPipeline):
            raise TypeError("sid_pipeline must implement ISidPipeline")


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
    max_rows: int | None = None,
) -> dict[str, Any]:
    if not run_ranking and not run_generation:
        raise ValueError("at least one evaluation stage must run")
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
        **dict(config.tracking.tags),
    }
    with tracker.start_run(run_name, tags=tags):
        tracker.log_params(
            flatten_parameters(
                {
                    "dataset": asdict(config.dataset),
                    "training": asdict(config.training),
                    "enabled_stages": list(enabled_stages),
                    "max_rows": max_rows,
                }
            )
        )
        return run_sid_experiment(
            SidExperimentConfig(
                dataset=dataset,
                sid_pipeline=config.sid_pipeline,
                enabled_stages=enabled_stages,
                reports_dir=config.reports_dir,
                core_min_interactions_per_item=(
                    config.dataset.core_min_interactions_per_item
                ),
                history_days=config.dataset.history_days,
                test_days=config.dataset.test_days,
                timestamp_unit_seconds=config.dataset.timestamp_unit_seconds,
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
        help="Python config path, optionally followed by :attribute.",
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
    args = build_parser().parse_args(argv)
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
        run_ranking=not args.skip_ranking,
        run_generation=not args.skip_generation,
        max_rows=args.max_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
