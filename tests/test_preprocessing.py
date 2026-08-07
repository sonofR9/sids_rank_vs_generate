from __future__ import annotations

import unittest

import polars as pl

from sids_rank_vs_generate.data import (
    PreprocessingConfig,
    build_histories,
    interaction_fingerprint,
    prepare_interactions,
)


DAY = 24 * 60 * 60


def sample_interactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1, 2, 1, 2, 1, 2, 1, 2],
            "item_id": [10, 20, 20, 10, 10, 20, 30, 30],
            "timestamp": [
                0,
                1,
                100,
                200,
                2 * DAY,
                2 * DAY + 1,
                3 * DAY,
                4 * DAY,
            ],
        }
    )


class PreprocessingTest(unittest.TestCase):
    def test_dense_mapping_and_chronological_split(self) -> None:
        interactions = pl.concat(
            [
                sample_interactions(),
                pl.DataFrame(
                    {
                        "uid": [1, 2],
                        "item_id": [10, 20],
                        "timestamp": [3 * DAY + 1, 4 * DAY + 1],
                    }
                ),
            ]
        )
        prepared = prepare_interactions(
            interactions,
            PreprocessingConfig(
                core_min_interactions_per_item=2,
                test_days=1,
                catalog_from_fit=True,
            ),
        )

        # Айтем 30 встречается только после fit и не входит в каталог
        self.assertEqual(prepared.catalog_size, 2)
        self.assertEqual(set(prepared.data["item_id"].to_list()), {0, 1})
        self.assertEqual(prepared.data.height, 8)
        self.assertEqual(prepared.train.height + prepared.test.height, 8)
        self.assertLess(
            prepared.train["timestamp"].max(),
            prepared.test["timestamp"].min(),
        )
        self.assertLess(
            prepared.fit["timestamp"].max(),
            prepared.validation["timestamp"].min(),
        )

    def test_histories_are_sorted(self) -> None:
        prepared = prepare_interactions(
            sample_interactions(),
            PreprocessingConfig(2, 1),
        )
        histories = build_histories(prepared.train)
        self.assertEqual(histories[1], [0, 1, 0])
        self.assertEqual(histories[2], [1, 0, 1])

    def test_yambda_five_second_timestamp_bins(self) -> None:
        interactions = pl.DataFrame(
            {
                "uid": [1, 1, 1],
                "item_id": [10, 10, 10],
                "timestamp": [0, DAY // 5, 2 * DAY // 5],
            }
        )
        prepared = prepare_interactions(
            interactions,
            PreprocessingConfig(
                core_min_interactions_per_item=1,
                test_days=1,
                timestamp_unit_seconds=5,
            ),
        )

        self.assertEqual(prepared.test_start_timestamp, DAY // 5)
        self.assertEqual(prepared.train.height, 1)
        self.assertEqual(prepared.test.height, 2)

    def test_fit_fingerprint_is_order_invariant_and_data_sensitive(self) -> None:
        frame = sample_interactions()
        first = interaction_fingerprint(frame)
        second = interaction_fingerprint(frame.reverse())
        changed = interaction_fingerprint(
            frame.with_columns((pl.col("timestamp") + 1).alias("timestamp"))
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
