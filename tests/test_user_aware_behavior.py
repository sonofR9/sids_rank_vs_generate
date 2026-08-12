from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.data.preprocessing import PreparedInteractions
from sids_rank_vs_generate.data.user_aware_behavior import (
    load_fit_user_aware_behavior,
    user_aware_behavior_fingerprint,
)


def prepared_interactions() -> PreparedInteractions:
    data = pl.DataFrame(
        {
            "uid": [1, 2, 1, 2],
            "item_id": [0, 1, 0, 1],
            "timestamp": [10, 20, 100, 110],
        }
    )
    fit = data.filter(pl.col("timestamp") < 100)
    validation = data.filter(pl.col("timestamp") == 100)
    test = data.filter(pl.col("timestamp") > 100)
    return PreparedInteractions(
        data=data,
        train=pl.concat([fit, validation]),
        test=test,
        item_mapping=pl.DataFrame(
            {
                "original_item_id": [10, 20],
                "item_id": pl.Series([0, 1], dtype=pl.Int64),
            }
        ),
        catalog_size=2,
        test_start_timestamp=110,
        timestamp_unit_seconds=1.0,
        fit=fit,
        validation=validation,
        validation_start_timestamp=100,
    )


def behavior_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [8, 3, 1, 2, 4, 5, 6, 7],
            "item_id": [10, 10, 20, 10, 20, 99, 10, 10],
            "timestamp": [5, 80, 20, 70, 90, 95, 100, 60],
            "is_like": [0, 1, 0, 0, 1, 1, 0, 1],
            "unused_lag_feature": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )


class UserAwareBehaviorTest(unittest.TestCase):
    def _write(self, directory: str, frame: pl.DataFrame, name: str = "rows.parquet") -> Path:
        path = Path(directory) / name
        frame.write_parquet(path)
        return path

    def test_filters_before_limit_maps_catalog_and_builds_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, behavior_frame())
            result = load_fit_user_aware_behavior(
                path,
                prepared_interactions(),
                max_rows=3,
            )

        self.assertEqual(result.frame.columns, ["uid", "item_id", "timestamp", "target"])
        self.assertEqual(result.frame["timestamp"].to_list(), [70, 80, 90])
        self.assertEqual(result.frame["item_id"].to_list(), [0, 0, 1])
        self.assertEqual(result.frame.schema["target"], pl.Float32)
        self.assertTrue((result.frame["timestamp"] < 100).all())
        self.assertEqual(result.summary.rows, 3)
        self.assertEqual(result.summary.users, 3)
        self.assertEqual(result.summary.pairs, 3)
        self.assertAlmostEqual(result.summary.positive_rate, 2 / 3)
        self.assertEqual(result.summary.min_timestamp, 70)
        self.assertEqual(result.summary.max_timestamp, 90)
        self.assertEqual(result.summary.fit_start_timestamp, 10)
        self.assertEqual(result.summary.fit_end_timestamp_exclusive, 100)
        self.assertEqual(len(result.summary.fingerprint), 64)

    def test_uses_the_same_lower_time_boundary_as_prepared_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, behavior_frame())
            result = load_fit_user_aware_behavior(path, prepared_interactions())

        self.assertNotIn(5, result.frame["timestamp"].to_list())
        self.assertGreaterEqual(
            result.summary.min_timestamp,
            result.summary.fit_start_timestamp,
        )

    def test_fingerprint_is_order_invariant_and_includes_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = self._write(directory, behavior_frame(), "first.parquet")
            second_path = self._write(
                directory,
                behavior_frame().reverse(),
                "second.parquet",
            )
            first = load_fit_user_aware_behavior(first_path, prepared_interactions())
            second = load_fit_user_aware_behavior(second_path, prepared_interactions())

        self.assertEqual(first.summary.fingerprint, second.summary.fingerprint)
        changed = first.frame.with_columns((1.0 - pl.col("target")).alias("target"))
        self.assertNotEqual(
            first.summary.fingerprint,
            user_aware_behavior_fingerprint(changed),
        )

    def test_rejects_nulls_in_fit_rows(self) -> None:
        frame = behavior_frame().with_columns(
            pl.when(pl.col("timestamp") == 70)
            .then(None)
            .otherwise(pl.col("is_like"))
            .alias("is_like")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, frame)
            with self.assertRaisesRegex(ValueError, "contain nulls"):
                load_fit_user_aware_behavior(path, prepared_interactions())

    def test_rejects_null_timestamps(self) -> None:
        frame = behavior_frame().with_columns(
            pl.when(pl.col("uid") == 6)
            .then(None)
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, frame)
            with self.assertRaisesRegex(ValueError, "timestamp contains nulls"):
                load_fit_user_aware_behavior(path, prepared_interactions())

    def test_rejects_empty_catalog_overlap(self) -> None:
        frame = behavior_frame().with_columns(pl.lit(99).alias("item_id"))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, frame)
            with self.assertRaisesRegex(ValueError, "no fit behavior rows"):
                load_fit_user_aware_behavior(path, prepared_interactions())

    def test_rejects_single_label(self) -> None:
        frame = behavior_frame().with_columns(pl.lit(1).alias("is_like"))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, frame)
            with self.assertRaisesRegex(ValueError, "both labels"):
                load_fit_user_aware_behavior(path, prepared_interactions())

    def test_rejects_missing_columns_and_nonpositive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, behavior_frame().drop("is_like"))
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_fit_user_aware_behavior(path, prepared_interactions())
            with self.assertRaisesRegex(ValueError, "max_rows"):
                load_fit_user_aware_behavior(path, prepared_interactions(), max_rows=0)


if __name__ == "__main__":
    unittest.main()
