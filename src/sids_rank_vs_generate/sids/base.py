from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..tracking import ExperimentTracker, NullTracker


@dataclass(frozen=True)
class SidTrainingData:
    """Inputs available to global SID pipelines at training time.

    ``train_interactions`` is cut strictly before ``train_end_timestamp``.  The
    explicit boundary lets collaborative pipelines validate the no-future-data
    contract instead of relying on experiment configuration conventions.
    """

    item_features: pl.DataFrame
    train_interactions: pl.DataFrame
    train_end_timestamp: Any

    def __post_init__(self) -> None:
        if self.item_features.is_empty():
            raise ValueError("item features must not be empty")
        if "item_id" not in self.item_features.columns:
            raise ValueError("item features must contain item_id")
        if self.item_features["item_id"].null_count():
            raise ValueError("item features contain null item_id values")
        if self.item_features["item_id"].n_unique() != self.item_features.height:
            raise ValueError("item features must contain one row per item_id")

        required = {"uid", "item_id", "timestamp"}
        missing = required - set(self.train_interactions.columns)
        if missing:
            raise ValueError(
                "train interactions are missing columns: "
                + ", ".join(sorted(missing))
            )
        if any(
            self.train_interactions[column].null_count()
            for column in required
        ):
            raise ValueError("train interactions contain nulls")
        if (
            not self.train_interactions.is_empty()
            and self.train_interactions["timestamp"].max()
            >= self.train_end_timestamp
        ):
            raise ValueError(
                "train interactions must be strictly earlier than "
                "train_end_timestamp"
            )
        catalog = set(self.item_features["item_id"].to_list())
        unknown_items = set(
            self.train_interactions["item_id"].unique().to_list()
        ) - catalog
        if unknown_items:
            raise ValueError("train interactions contain items outside the catalog")


class ISidPipeline(ABC):
    """Base interface for all SIDs.

    Defines the common train + update pipeline.
    Concrete SIDs derive from this and implement the abstract methods.
    """

    def set_tracker(self, tracker: ExperimentTracker) -> None:
        """Attach the tracker used by train loops without changing the SID API."""

        self._tracker = tracker

    @property
    def tracker(self) -> ExperimentTracker:
        return getattr(self, "_tracker", NullTracker())

    @abstractmethod
    def train(self, df: pl.DataFrame) -> None:
        """Train on dense item_id plus all columns from the item-feature parquet."""

    def train_with_context(self, data: SidTrainingData) -> None:
        """Train with the shared temporal context.

        Content-only pipelines keep their existing ``train(DataFrame)`` API.
        Collaborative pipelines override this method to consume train-only
        interactions while remaining compatible with the common runner.
        """

        self.train(data.item_features)

    @abstractmethod
    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per item_id and integer sid_0, ..., sid_n columns."""
