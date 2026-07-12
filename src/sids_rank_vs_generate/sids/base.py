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
        """Train the SID on the given DataFrame."""

    @abstractmethod
    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run inference and append SID output columns to the given DataFrame."""
