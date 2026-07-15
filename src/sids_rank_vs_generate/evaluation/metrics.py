from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence


def get_ranking_metrics(
    targets: Sequence[int],
    candidates: Sequence[int],
    topk: int,
) -> dict[str, float]:
    if topk <= 0:
        raise ValueError("topk must be positive")

    target_set = set(targets)
    if not target_set:
        return {"hitrate": 0.0, "recall": 0.0, "ndcg": 0.0}

    relevance = [
        int(item_id in target_set) for item_id in candidates[:topk]
    ]
    hits = sum(relevance)
    dcg = sum(
        relevant / math.log2(rank + 2)
        for rank, relevant in enumerate(relevance)
    )
    ideal_hits = min(len(target_set), topk)
    idcg = sum(
        1.0 / math.log2(rank + 2) for rank in range(ideal_hits)
    )
    return {
        "hitrate": float(hits > 0),
        "recall": hits / len(target_set),
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def evaluate_ranking(
    targets: Mapping[Hashable, Sequence[int]],
    candidates: Mapping[Hashable, Sequence[int]],
    catalog_size: int,
    topk: int,
) -> dict[str, float]:
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    if not targets:
        return {
            "hitrate": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "coverage": 0.0,
        }

    per_user = [
        get_ranking_metrics(user_targets, candidates.get(uid, []), topk)
        for uid, user_targets in targets.items()
    ]
    recommended = {
        item_id
        for uid in targets
        for item_id in candidates.get(uid, [])[:topk]
    }
    user_count = len(per_user)
    return {
        metric: sum(row[metric] for row in per_user) / user_count
        for metric in ("hitrate", "recall", "ndcg")
    } | {"coverage": len(recommended) / catalog_size}
