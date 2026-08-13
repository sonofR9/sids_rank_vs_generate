from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .data import (
    PreparedInteractions,
    PreprocessingConfig,
    load_interactions,
    prepare_interactions,
)
from .data.protocols import THREE_WAY_PROTOCOL_ID
from .reproducibility import SID_EXPERIMENT_SEED, seed_everything


@dataclass(frozen=True)
class PipelineConfig:
    interactions_path: Path
    core_min_interactions_per_item: int = 5
    history_days: float | None = None
    test_days: float = 7.0
    timestamp_unit_seconds: float = 1.0
    validation_fraction: float = 0.1
    catalog_from_fit: bool = False
    latest_interactions: int | None = None
    protocol_id: str = THREE_WAY_PROTOCOL_ID

    @property
    def preprocessing(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            core_min_interactions_per_item=self.core_min_interactions_per_item,
            history_days=self.history_days,
            test_days=self.test_days,
            timestamp_unit_seconds=self.timestamp_unit_seconds,
            validation_fraction=self.validation_fraction,
            catalog_from_fit=self.catalog_from_fit,
            protocol_id=self.protocol_id,
        )


def run_pipeline(
    config: PipelineConfig,
    eligible_items: pl.DataFrame | None = None,
) -> PreparedInteractions:
    """Run the deterministic common part used by every experiment."""

    seed_everything(SID_EXPERIMENT_SEED)
    interactions = load_interactions(
        config.interactions_path,
        latest_interactions=config.latest_interactions,
    )
    if eligible_items is not None:
        if "item_id" not in eligible_items.columns:
            raise ValueError("eligible_items must contain item_id")
        eligible_item_ids = eligible_items.select("item_id").unique()
        if eligible_item_ids.is_empty():
            raise ValueError("eligible_items must not be empty")
        interactions = interactions.join(
            eligible_item_ids,
            on="item_id",
            how="inner",
        )
        if interactions.is_empty():
            raise ValueError("no interactions match the eligible item catalog")
    return prepare_interactions(interactions, config=config.preprocessing)
