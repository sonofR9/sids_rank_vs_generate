from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl

from ..tracking import ExperimentTracker
from .user_aware import (
    IUserAwareSidPipeline,
    UserAwareSidTrainingData,
    user_sid_columns,
    validate_user_sid_output,
)


@dataclass(frozen=True)
class LevelPermutationConfig:
    """Настройки перестановки SID уровней"""

    codebook_sizes: tuple[int, ...]
    level_permutation: tuple[int, ...]
    base_pipeline: str


class LevelPermutationUserAwareSidPipeline(IUserAwareSidPipeline):
    """Меняет порядок usid после присвоения кодов"""

    def __init__(
        self,
        pipeline: IUserAwareSidPipeline,
        *,
        codebook_sizes: Sequence[int],
        level_permutation: Sequence[int],
    ) -> None:
        if not isinstance(pipeline, IUserAwareSidPipeline):
            raise TypeError("pipeline must implement IUserAwareSidPipeline")
        source_sizes = tuple(int(size) for size in codebook_sizes)
        permutation = tuple(int(level) for level in level_permutation)
        if not source_sizes or any(size <= 0 for size in source_sizes):
            raise ValueError("codebook_sizes must contain positive values")
        if tuple(sorted(permutation)) != tuple(range(len(source_sizes))):
            raise ValueError("level_permutation must contain every level once")

        self.pipeline = pipeline
        self.source_codebook_sizes = source_sizes
        self.config = LevelPermutationConfig(
            codebook_sizes=tuple(source_sizes[level] for level in permutation),
            level_permutation=permutation,
            base_pipeline=(
                f"{type(pipeline).__module__}:{type(pipeline).__name__}"
            ),
        )

    @property
    def requires_behavior_interactions(self) -> bool:
        return self.pipeline.requires_behavior_interactions

    def set_tracker(self, tracker: ExperimentTracker) -> None:
        super().set_tracker(tracker)
        self.pipeline.set_tracker(tracker)

    def train(self, data: UserAwareSidTrainingData) -> None:
        self.pipeline.train(data)

    def update(self, pairs: pl.DataFrame) -> pl.DataFrame:
        assigned = self.pipeline.update(pairs)
        columns = user_sid_columns(assigned)
        if len(columns) != len(self.config.level_permutation):
            raise ValueError(
                "base user SID level count does not match level_permutation"
            )
        output = assigned.select(
            "uid",
            "item_id",
            *(
                pl.col(columns[source_level]).alias(f"usid_{target_level}")
                for target_level, source_level in enumerate(
                    self.config.level_permutation
                )
            ),
        )
        validate_user_sid_output(pairs, output)
        return output

    def user_features(self, users: pl.DataFrame) -> pl.DataFrame:
        return self.pipeline.user_features(users)

    def summary(self) -> dict[str, Any]:
        base_summary = getattr(self.pipeline, "summary", None)
        return {
            "level_permutation": list(self.config.level_permutation),
            "source_codebook_sizes": list(self.source_codebook_sizes),
            "output_codebook_sizes": list(self.config.codebook_sizes),
            "base": (
                deepcopy(base_summary()) if callable(base_summary) else None
            ),
        }
