from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data import (
    PreparedInteractions,
    PreprocessingConfig,
    load_and_prepare_interactions,
)
from .reproducibility import seed_everything


@dataclass(frozen=True)
class PipelineConfig:
    interactions_path: Path
    core_min_interactions_per_item: int = 5
    test_days: float = 7.0
    seed: int = 42

    @property
    def preprocessing(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            core_min_interactions_per_item=self.core_min_interactions_per_item,
            test_days=self.test_days,
        )


def run_pipeline(config: PipelineConfig) -> PreparedInteractions:
    """Run the deterministic common part used by every experiment."""

    seed_everything(config.seed)
    return load_and_prepare_interactions(
        config.interactions_path,
        config=config.preprocessing,
    )
