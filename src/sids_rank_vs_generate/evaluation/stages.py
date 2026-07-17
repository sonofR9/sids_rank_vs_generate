from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import polars as pl

from ..data import PreparedInteractions


@dataclass(frozen=True)
class EvaluationContext:
    """Common immutable inputs supplied to every downstream evaluation stage."""

    prepared: PreparedInteractions
    items_with_sids: pl.DataFrame
    seed: int


class IEvaluationStage(ABC):
    """Interface implemented by ranking and candidate-generation evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable stage name used in metrics and reports."""

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> Mapping[str, Any]:
        """Train/evaluate the stage and return all inexpensive metrics."""
