from __future__ import annotations

from collections import defaultdict
from typing import Any

import polars as pl

from ..sids.user_aware import user_sid_columns, validate_user_sid_output


def calculate_user_sid_statistics(frame: pl.DataFrame) -> dict[str, Any]:
    """Статистики персональных SID для уникальных пар"""

    columns = validate_user_sid_output(frame.select("uid", "item_id"), frame)
    paths: dict[tuple[int, ...], int] = defaultdict(int)
    user_paths: dict[object, set[tuple[int, ...]]] = defaultdict(set)
    user_pairs: dict[object, int] = defaultdict(int)
    item_paths: dict[object, set[tuple[int, ...]]] = defaultdict(set)

    for row in frame.select("uid", "item_id", *columns).iter_rows():
        uid, item_id, *codes = row
        path = tuple(int(code) for code in codes)
        paths[path] += 1
        user_paths[uid].add(path)
        user_pairs[uid] += 1
        item_paths[item_id].add(path)

    pair_count = frame.height
    unique_paths = len(paths)
    per_user_rates = sorted(
        len(user_paths[uid]) / user_pairs[uid] for uid in user_pairs
    )
    per_item_variation = sorted(len(values) for values in item_paths.values())

    def percentile(values: list[float] | list[int], fraction: float) -> float:
        index = round((len(values) - 1) * fraction)
        return float(values[index])

    return {
        "pair_count": pair_count,
        "user_count": len(user_pairs),
        "item_count": len(item_paths),
        "columns": user_sid_columns(frame),
        "unique_paths": unique_paths,
        "independent_code_rate": unique_paths / pair_count,
        "collision_rate": 1.0 - unique_paths / pair_count,
        "max_collision": max(paths.values()),
        "levels": {
            column: {
                "used_codes": frame[column].n_unique(),
                "min_code": int(frame[column].min()),
                "max_code": int(frame[column].max()),
            }
            for column in columns
        },
        "per_user_unique_path_rate": {
            "mean": sum(per_user_rates) / len(per_user_rates),
            "p50": percentile(per_user_rates, 0.5),
            "p10": percentile(per_user_rates, 0.1),
            "min": per_user_rates[0],
        },
        "same_item_cross_user_variation": {
            "mean_unique_paths": (sum(per_item_variation) / len(per_item_variation)),
            "max_unique_paths": per_item_variation[-1],
            "items_with_multiple_paths_rate": sum(
                count > 1 for count in per_item_variation
            )
            / len(per_item_variation),
        },
    }
