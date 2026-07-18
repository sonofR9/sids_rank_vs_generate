from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class ISidPipeline(ABC):
    """Base interface for all SIDs.

    Defines the common train + update pipeline.
    Concrete SIDs derive from this and implement the abstract methods.
    """

    @abstractmethod
    def train(self, df: pl.DataFrame) -> None:
        """Train on dense item_id plus all columns from the item-feature parquet."""

    @abstractmethod
    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per item_id and integer sid_0, ..., sid_n columns."""
