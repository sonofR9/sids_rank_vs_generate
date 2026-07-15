from __future__ import annotations

import unittest

import polars as pl

from sids_rank_vs_generate.data import (
    PreprocessingConfig,
    build_histories,
    prepare_interactions,
)


DAY = 24 * 60 * 60


def sample_interactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1, 2, 1, 2, 1, 2, 1, 2],
            "item_id": [10, 20, 20, 10, 10, 20, 30, 30],
            "timestamp": [0, 1, 100, 200, 2 * DAY, 2 * DAY + 1, 3 * DAY, 4 * DAY],
        }
    )


class PreprocessingTest(unittest.TestCase):
    def test_dense_mapping_and_chronological_split(self) -> None:
        prepared = prepare_interactions(
            sample_interactions(),
            PreprocessingConfig(
                core_min_interactions_per_item=2,
                test_days=1,
            ),
        )

        self.assertEqual(prepared.catalog_size, 3)
        self.assertEqual(set(prepared.data["item_id"].to_list()), {0, 1, 2})
        self.assertEqual(prepared.data.height, 8)
        self.assertEqual(prepared.train.height + prepared.test.height, 8)
        self.assertLess(
            prepared.train["timestamp"].max(),
            prepared.test["timestamp"].min(),
        )

    def test_histories_are_sorted(self) -> None:
        prepared = prepare_interactions(
            sample_interactions(),
            PreprocessingConfig(2, 1),
        )
        histories = build_histories(prepared.train)
        self.assertEqual(histories[1], [0, 1, 0])
        self.assertEqual(histories[2], [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
