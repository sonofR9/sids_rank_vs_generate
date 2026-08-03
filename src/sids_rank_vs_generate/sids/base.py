from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from ..tracking import ExperimentTracker, NullTracker


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

    @abstractmethod
    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per item_id and integer sid_0, ..., sid_n columns."""
