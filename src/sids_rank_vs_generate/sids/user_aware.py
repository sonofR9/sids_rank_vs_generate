from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..tracking import ExperimentTracker, NullTracker


PAIR_COLUMNS = ("uid", "item_id")
USER_SID_PREFIX = "usid_"


def user_sid_columns(frame: pl.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith(USER_SID_PREFIX)
        and column[len(USER_SID_PREFIX) :].isdigit()
    ]
    return sorted(
        columns,
        key=lambda column: int(column[len(USER_SID_PREFIX) :]),
    )


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    description: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{description} is missing columns: {', '.join(sorted(missing))}"
        )


@dataclass(frozen=True)
class UserAwareSidTrainingData:
    """Данные для обучения персональных SID без утечки"""

    fit_interactions: pl.DataFrame
    item_features: pl.DataFrame
    global_sids: pl.DataFrame
    fit_end_timestamp: Any
    fit_behavior_interactions: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        _require_columns(
            self.fit_interactions,
            {"uid", "item_id", "timestamp"},
            "fit interactions",
        )
        _require_columns(
            self.item_features,
            {"item_id"},
            "item features",
        )
        _require_columns(
            self.global_sids,
            {"item_id"},
            "global SIDs",
        )
        if self.fit_interactions.is_empty():
            raise ValueError("fit interactions must not be empty")
        if self.item_features.is_empty():
            raise ValueError("item features must not be empty")
        if self.global_sids.is_empty():
            raise ValueError("global SIDs must not be empty")
        if self.fit_interactions["timestamp"].max() >= self.fit_end_timestamp:
            raise ValueError(
                "fit interactions must be strictly earlier than fit_end_timestamp"
            )
        for description, frame in (
            ("item features", self.item_features),
            ("global SIDs", self.global_sids),
        ):
            if frame["item_id"].null_count():
                raise ValueError(f"{description} contain null item_id values")
            if frame["item_id"].n_unique() != frame.height:
                raise ValueError(f"{description} must contain one row per item_id")
        global_sid_columns = [
            column
            for column in self.global_sids.columns
            if column.startswith("sid_") and column[4:].isdigit()
        ]
        if not global_sid_columns:
            raise ValueError("global SIDs must contain sid_<level> columns")
        feature_items = set(self.item_features["item_id"].to_list())
        sid_items = set(self.global_sids["item_id"].to_list())
        if feature_items != sid_items:
            raise ValueError(
                "item features and global SIDs must cover the same catalog"
            )
        unknown_fit_items = (
            set(self.fit_interactions["item_id"].unique().to_list()) - feature_items
        )
        if unknown_fit_items:
            raise ValueError("fit interactions contain items outside the catalog")
        behavior = self.fit_behavior_interactions
        if behavior is not None:
            _require_columns(
                behavior,
                {"uid", "item_id", "timestamp", "target"},
                "fit behavior interactions",
            )
            if behavior.is_empty():
                raise ValueError("fit behavior interactions must not be empty")
            if any(behavior[column].null_count() for column in behavior.columns):
                raise ValueError("fit behavior interactions contain nulls")
            if behavior["timestamp"].max() >= self.fit_end_timestamp:
                raise ValueError(
                    "fit behavior interactions must be strictly earlier than "
                    "fit_end_timestamp"
                )
            labels = set(behavior["target"].unique().to_list())
            if labels != {0.0, 1.0}:
                raise ValueError("fit behavior interactions must contain both labels")
            unknown_behavior_items = (
                set(behavior["item_id"].unique().to_list()) - feature_items
            )
            if unknown_behavior_items:
                raise ValueError(
                    "fit behavior interactions contain items outside the catalog"
                )


class IUserAwareSidPipeline(ABC):
    """Интерфейс построения персональных SID"""

    def set_tracker(self, tracker: ExperimentTracker) -> None:
        """Подключить трекер обучения"""

        self._tracker = tracker

    @property
    def tracker(self) -> ExperimentTracker:
        return getattr(self, "_tracker", NullTracker())

    @property
    def requires_behavior_interactions(self) -> bool:
        """Нужны ли pipeline поведенческие fit-метки"""

        return False

    @abstractmethod
    def train(self, data: UserAwareSidTrainingData) -> None:
        """Обучить профили и кодбуки только на fit"""

    @abstractmethod
    def update(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """Построить SID для уникальных пар пользователя и айтема"""

    @abstractmethod
    def user_features(self, users: pl.DataFrame) -> pl.DataFrame:
        """Вернуть плотные признаки пользователей"""


def validate_user_sid_output(
    source_pairs: pl.DataFrame,
    output: pl.DataFrame,
) -> list[str]:
    _require_columns(source_pairs, set(PAIR_COLUMNS), "source pairs")
    _require_columns(output, set(PAIR_COLUMNS), "user SID output")
    if source_pairs.select(*PAIR_COLUMNS).unique().height != source_pairs.height:
        raise ValueError("source pairs must contain unique (uid, item_id) rows")
    if output.height != source_pairs.height:
        raise ValueError("user SID output must contain one row per source pair")
    if any(output[column].null_count() for column in PAIR_COLUMNS):
        raise ValueError("user SID output contains null pair identifiers")
    if output.select(*PAIR_COLUMNS).unique().height != output.height:
        raise ValueError("user SID output contains duplicate pairs")

    source_keys = set(source_pairs.select(*PAIR_COLUMNS).iter_rows())
    output_keys = set(output.select(*PAIR_COLUMNS).iter_rows())
    if source_keys != output_keys:
        raise ValueError("user SID output pairs do not match source pairs")

    columns = user_sid_columns(output)
    if not columns:
        raise ValueError("user SID output must contain usid_<level> columns")
    for column in columns:
        if output[column].null_count():
            raise ValueError(f"user SID output column {column} contains nulls")
        if not output.schema[column].is_integer():
            raise TypeError(f"user SID output column {column} must be integer")
    return columns
