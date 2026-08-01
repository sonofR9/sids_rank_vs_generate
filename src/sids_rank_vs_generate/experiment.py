from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from .datasets import DatasetFiles, resolve_dataset
from .evaluation.generation_stage import GenerationConfig, GenerationStage
from .evaluation.ranking_stage import RankingConfig, RankingStage
from .sid_experiment import SidExperimentConfig, run_sid_experiment
from .sids import ISidPipeline


@dataclass(frozen=True)
class YambdaDatasetConfig:
    size: str = "50m"
    interactions_path: Path | None = None
    embeddings_path: Path | None = None
    source_embeddings_path: Path | None = None
    core_min_interactions_per_item: int = 5
    test_days: float = 7.0
    timestamp_unit_seconds: float = 5.0
    latest_interactions: int | None = None

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
class ExperimentConfig:
    sid_pipeline: ISidPipeline
    dataset: YambdaDatasetConfig = YambdaDatasetConfig()
    ranking: RankingConfig | None = None
    generation: GenerationConfig | None = None
    reports_dir: Path = Path("reports/experiments")
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.sid_pipeline, ISidPipeline):
            raise TypeError("sid_pipeline must implement ISidPipeline")
        if self.ranking is None and self.generation is None:
            raise ValueError("enable at least one of ranking or generation")


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


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    dataset = config.dataset.resolve()
    stages = {}
    if config.ranking is not None:
        stages["ranking"] = RankingStage(config.ranking)
    if config.generation is not None:
        stages["generation"] = GenerationStage(config.generation)

    def resolve_stage(name: str):
        return stages[name]

    return run_sid_experiment(
        SidExperimentConfig(
            dataset=dataset,
            sid_pipeline=config.sid_pipeline,
            enabled_stages=tuple(stages),
            reports_dir=config.reports_dir,
            core_min_interactions_per_item=(
                config.dataset.core_min_interactions_per_item
            ),
            test_days=config.dataset.test_days,
            timestamp_unit_seconds=config.dataset.timestamp_unit_seconds,
            latest_interactions=config.dataset.latest_interactions,
            seed=config.seed,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    result = run_experiment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
