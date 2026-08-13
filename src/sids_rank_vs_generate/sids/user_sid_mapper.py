from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import polars as pl

from .collision_resolution import get_sid_cols
from .user_aware import user_sid_columns


SidPath = tuple[int, ...]
MapperKey = tuple[object, SidPath]


def _mapping_counters(
    frame: pl.DataFrame,
) -> dict[MapperKey, Counter[SidPath]]:
    user_columns = user_sid_columns(frame)
    global_columns = get_sid_cols(frame)
    if not user_columns or not global_columns:
        raise ValueError("mapper diagnostics require usid_* and sid_* columns")
    if frame.is_empty():
        raise ValueError("mapper diagnostics require non-empty data")

    counters: dict[MapperKey, Counter[SidPath]] = defaultdict(Counter)
    columns = ["uid", *user_columns, *global_columns]
    split = 1 + len(user_columns)
    for row in frame.select(*columns).iter_rows():
        key = (row[0], tuple(int(code) for code in row[1:split]))
        target = tuple(int(code) for code in row[split:])
        counters[key][target] += 1
    return counters


def mapping_ambiguity_metrics(
    frame: pl.DataFrame,
    *,
    top_k: int,
) -> dict[str, int | float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counters = _mapping_counters(frame)
    examples = sum(sum(counter.values()) for counter in counters.values())
    target_counts = [len(counter) for counter in counters.values()]
    ambiguous = [counter for counter in counters.values() if len(counter) > 1]
    ambiguous_examples = sum(sum(counter.values()) for counter in ambiguous)
    top1_hits = sum(max(counter.values()) for counter in counters.values())
    topk_hits = sum(
        sum(sorted(counter.values(), reverse=True)[:top_k])
        for counter in counters.values()
    )
    ordered_counts = sorted(target_counts)
    p90_index = int(0.9 * (len(ordered_counts) - 1))
    return {
        "keys": len(counters),
        "examples": examples,
        "ambiguous_keys": len(ambiguous),
        "ambiguous_key_rate": len(ambiguous) / len(counters),
        "ambiguous_example_rate": ambiguous_examples / examples,
        "mean_targets_per_key": sum(target_counts) / len(target_counts),
        "p90_targets_per_key": ordered_counts[p90_index],
        "max_targets_per_key": max(target_counts),
        "oracle_top_1_accuracy": top1_hits / examples,
        f"oracle_top_{top_k}_accuracy": topk_hits / examples,
    }


def mapping_fit_support_metrics(
    frame: pl.DataFrame,
    examples: list[dict],
) -> dict[str, int | float | None]:
    if not examples:
        raise ValueError("mapper diagnostics require evaluation examples")
    counters = _mapping_counters(frame)
    seen_examples = 0
    supported_examples = 0
    candidate_counts = []
    for example in examples:
        key = (
            example["uid"],
            tuple(int(code) for code in example["sid"]),
        )
        candidates = counters.get(key)
        if candidates is None:
            continue
        seen_examples += 1
        candidate_counts.append(len(candidates))
        target = tuple(int(code) for code in example["global_sid"])
        supported_examples += target in candidates
    count = len(examples)
    return {
        "examples": count,
        "seen_examples": seen_examples,
        "seen_example_rate": seen_examples / count,
        "target_support_rate": supported_examples / count,
        "target_support_given_seen": (
            supported_examples / seen_examples if seen_examples else None
        ),
        "mean_candidates_given_seen": (
            sum(candidate_counts) / len(candidate_counts)
            if candidate_counts
            else None
        ),
        "max_candidates_given_seen": max(candidate_counts, default=0),
    }


class IUserSidMapper(ABC):
    """Интерфейс перевода персональных SID в глобальные"""

    @abstractmethod
    def predict_topk(
        self,
        uid: object,
        user_sid: Iterable[int],
        k: int,
    ) -> list[SidPath]:
        """Вернуть до k глобальных SID по убыванию качества"""

    def predict_topk_batch(
        self,
        requests: list[tuple[object, Iterable[int]]],
        k: int,
    ) -> list[list[SidPath]]:
        return [self.predict_topk(uid, user_sid, k) for uid, user_sid in requests]

    def is_seen(self, uid: object, user_sid: Iterable[int]) -> bool:
        """Проверить наличие пары в обучении"""

        return False

    def evaluate_examples(self, examples: list[dict], k: int = 10) -> dict:
        if not examples:
            raise ValueError("mapper evaluation examples must not be empty")
        predictions = self.predict_topk_batch(
            [(example["uid"], example["sid"]) for example in examples],
            k,
        )
        exact = 0
        topk = 0
        token_correct = 0
        token_total = 0
        strata = {
            "seen": {"examples": 0, "exact": 0, "topk": 0},
            "unseen": {"examples": 0, "exact": 0, "topk": 0},
        }
        for example, ranked_paths in zip(examples, predictions, strict=True):
            if not ranked_paths:
                raise RuntimeError("mapper returned no global SID candidates")
            target = tuple(int(code) for code in example["global_sid"])
            exact += ranked_paths[0] == target
            topk += target in ranked_paths
            stratum = (
                "seen" if self.is_seen(example["uid"], example["sid"]) else "unseen"
            )
            strata[stratum]["examples"] += 1
            strata[stratum]["exact"] += ranked_paths[0] == target
            strata[stratum]["topk"] += target in ranked_paths
            token_correct += sum(
                predicted == expected
                for predicted, expected in zip(ranked_paths[0], target)
            )
            token_total += len(target)
        count = len(examples)
        result: dict[str, Any] = {
            "token_accuracy": token_correct / token_total,
            "exact_accuracy": exact / count,
            f"top_{k}_accuracy": topk / count,
        }
        for name, values in strata.items():
            stratum_count = values["examples"]
            result[name] = {
                "examples": stratum_count,
                "exact_accuracy": (
                    values["exact"] / stratum_count if stratum_count else None
                ),
                f"top_{k}_accuracy": (
                    values["topk"] / stratum_count if stratum_count else None
                ),
            }
        return result


class FrequencyUserSidMapper(IUserSidMapper):
    """Частотный маппер из персонального SID в глобальный"""

    def __init__(self) -> None:
        self._by_user: dict[tuple[object, SidPath], list[SidPath]] = {}
        self._by_user_sid: dict[SidPath, list[SidPath]] = {}
        self._global: list[SidPath] = []
        self.user_sid_columns: list[str] = []
        self.global_sid_columns: list[str] = []

    @staticmethod
    def _rank(counter: Counter[SidPath]) -> list[SidPath]:
        return [
            path
            for path, _ in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    def fit(self, frame: pl.DataFrame) -> None:
        self.user_sid_columns = user_sid_columns(frame)
        self.global_sid_columns = get_sid_cols(frame)
        if not self.user_sid_columns or not self.global_sid_columns:
            raise ValueError("mapper training requires usid_* and sid_* columns")
        if frame.is_empty():
            raise ValueError("mapper training data must not be empty")

        by_user: dict[tuple[object, SidPath], Counter[SidPath]] = defaultdict(Counter)
        by_user_sid: dict[SidPath, Counter[SidPath]] = defaultdict(Counter)
        global_counts: Counter[SidPath] = Counter()
        columns = [
            "uid",
            *self.user_sid_columns,
            *self.global_sid_columns,
        ]
        for row in frame.select(*columns).iter_rows():
            uid = row[0]
            split = 1 + len(self.user_sid_columns)
            user_sid = tuple(int(code) for code in row[1:split])
            global_sid = tuple(int(code) for code in row[split:])
            by_user[(uid, user_sid)][global_sid] += 1
            by_user_sid[user_sid][global_sid] += 1
            global_counts[global_sid] += 1

        self._by_user = {key: self._rank(counts) for key, counts in by_user.items()}
        self._by_user_sid = {
            key: self._rank(counts) for key, counts in by_user_sid.items()
        }
        self._global = self._rank(global_counts)

    def predict_topk(
        self,
        uid: object,
        user_sid: Iterable[int],
        k: int,
    ) -> list[SidPath]:
        if not self._global:
            raise RuntimeError("fit() must be called before predict_topk()")
        if k <= 0:
            raise ValueError("k must be positive")
        path = tuple(int(code) for code in user_sid)
        candidates = []
        seen = set()
        for tier in (
            self._by_user.get((uid, path), []),
            self._by_user_sid.get(path, []),
            self._global,
        ):
            for candidate in tier:
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
                if len(candidates) >= k:
                    return candidates
        return candidates

    def is_seen(self, uid: object, user_sid: Iterable[int]) -> bool:
        path = tuple(int(code) for code in user_sid)
        return (uid, path) in self._by_user
