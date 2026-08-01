from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from ..data import PreparedInteractions
from ..datasets import DatasetFiles


@dataclass(frozen=True)
class EvaluationContext:
    """Common immutable inputs supplied to every downstream evaluation stage."""

    prepared: PreparedInteractions
    items_with_sids: pl.DataFrame
    dataset: DatasetFiles
    seed: int
    artifact_dir: Path
    max_rows: int | None = None


class IEvaluationStage(ABC):
    """Interface implemented by ranking and candidate-generation evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable stage name used in metrics and reports."""

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> Mapping[str, Any]:
        """Train/evaluate the stage and return all inexpensive metrics."""


StageFactory = Callable[[], IEvaluationStage]

# Concrete implementations are connected here when their feature branches are
# merged. Experiments select stage names; they never provide implementation paths.
DEFAULT_STAGE_FACTORIES: dict[str, StageFactory] = {}


def create_evaluation_stage(name: str) -> IEvaluationStage:
    """Create one of the fixed evaluation stages used by the shared pipeline."""

    try:
        factory = DEFAULT_STAGE_FACTORIES[name]
    except KeyError as error:
        raise RuntimeError(
            f"evaluation stage {name!r} is not connected yet; merge its common "
            "implementation before enabling this stage"
        ) from error
    stage = factory()
    if not isinstance(stage, IEvaluationStage):
        raise TypeError(
            f"factory for {name!r} produced {type(stage).__name__}; "
            "expected IEvaluationStage"
        )
    if stage.name != name:
        raise ValueError(f"factory for {name!r} produced a stage named {stage.name!r}")
    return stage
