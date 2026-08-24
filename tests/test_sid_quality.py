from __future__ import annotations

import math
import unittest

import numpy as np
import polars as pl

from sids_rank_vs_generate.evaluation.sid_quality import (
    SID_QUALITY_SCHEMA_VERSION,
    UserCatalogQualityConfig,
    calculate_sid_quality,
    calculate_user_catalog_sid_quality,
    calculate_user_catalog_sid_quality_from_frame,
    select_stable_quality_users,
)


def sample_embeddings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": [0, 1, 2, 3, 4, 5],
            "normalized_embed": [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ],
        }
    )


class FakeUserSidPipeline:
    def update(self, pairs: pl.DataFrame) -> pl.DataFrame:
        codes = {
            1: [(0, 0), (0, 0), (0, 1), (1, 0)],
            2: [(0, 0), (0, 0), (1, 1), (1, 1)],
        }
        rows = []
        for uid, item_id in pairs.iter_rows():
            sid_0, sid_1 = codes[uid][item_id]
            rows.append((uid, item_id, sid_0, sid_1))
        return pl.DataFrame(
            rows,
            schema={
                "uid": pl.Int64,
                "item_id": pl.Int64,
                "usid_0": pl.Int32,
                "usid_1": pl.Int32,
            },
            orient="row",
        )


class ShuffledParityUserSidPipeline:
    def __init__(self) -> None:
        self.codes = {
            10: [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0), (1, 0)],
            20: [(0, 0), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)],
        }

    def update(self, pairs: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for uid, item_id in pairs.iter_rows():
            sid_0, sid_1 = self.codes[uid][item_id]
            rows.append((uid, item_id, sid_0, sid_1))
        return pl.DataFrame(
            rows,
            schema={
                "uid": pl.Int64,
                "item_id": pl.Int64,
                "usid_0": pl.Int32,
                "usid_1": pl.Int32,
            },
            orient="row",
        ).reverse()


def assert_nested_metrics_equal(
    case: unittest.TestCase,
    actual: object,
    expected: object,
) -> None:
    if isinstance(expected, dict):
        case.assertIsInstance(actual, dict)
        assert isinstance(actual, dict)
        case.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            assert_nested_metrics_equal(case, actual[key], value)
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        case.assertAlmostEqual(float(actual), float(expected), places=12)
        return
    case.assertEqual(actual, expected)


class SidQualityTest(unittest.TestCase):
    def test_stable_user_sample_is_order_independent_and_fingerprinted(self) -> None:
        config = UserCatalogQualityConfig(enabled=True, max_users=3, seed=42)
        users = pl.DataFrame({"uid": [5, 1, 4, 2, 3, 1]})

        first, first_summary = select_stable_quality_users(users, config)
        second, second_summary = select_stable_quality_users(
            users.reverse(),
            config,
        )

        self.assertEqual(first["uid"].to_list(), second["uid"].to_list())
        self.assertEqual(first.height, 3)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary["eligible_user_count"], 5)
        self.assertEqual(first_summary["selected_user_count"], 3)
        self.assertEqual(len(first_summary["user_sample_fingerprint"]), 64)

    def test_global_quality_matches_main_exporter_recompute_sid_metrics(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3, 4, 5],
                "sid_0": pl.Series([0, 0, 0, 1, 1, 3], dtype=pl.Int32),
                "sid_1": pl.Series([0, 0, 1, 1, 1, 0], dtype=pl.Int32),
            }
        )

        metrics = calculate_sid_quality(
            sids,
            (2, 2),
            sample_embeddings(),
        )

        counts = np.asarray([2.0, 1.0, 2.0])
        probabilities = counts / counts.sum()
        expected_entropy = float(
            -(probabilities * np.log(probabilities)).sum()
        )
        expected_gini = float(
            (2 * np.dot(np.arange(1, 4), np.sort(counts)) / (3 * counts.sum()))
            - 4 / 3
        )

        self.assertEqual(metrics["schema_version"], SID_QUALITY_SCHEMA_VERSION)
        self.assertAlmostEqual(metrics["valid_sid_rate"], 5 / 6)
        self.assertAlmostEqual(metrics["independent_code_rate"], 3 / 5)
        self.assertAlmostEqual(metrics["collision_rate"], 2 / 5)
        self.assertEqual(metrics["collision_percentiles"]["p95"], 2.0)
        self.assertEqual(metrics["max_collision"], 2)
        self.assertAlmostEqual(
            metrics["codebook_quality"][
                "codebook_utilization_rate_total"
            ],
            1.0,
        )
        self.assertAlmostEqual(metrics["code_gini"], expected_gini)
        self.assertAlmostEqual(metrics["code_entropy"], expected_entropy)
        self.assertAlmostEqual(
            metrics["code_entropy_normalized"],
            expected_entropy / math.log(4),
        )
        self.assertAlmostEqual(metrics["prefix_coherence_at_1"], -1 / 3)
        self.assertAlmostEqual(metrics["prefix_coherence_at_2"], 0.0)
        self.assertAlmostEqual(metrics["intra_code_similarity"], 0.0)

    def test_collision_p95_uses_only_collision_buckets_and_nearest(self) -> None:
        sizes = [1, 2, 3, 4, 5, 6, 100]
        paths = [path for path, size in enumerate(sizes) for _ in range(size)]
        sids = pl.DataFrame(
            {
                "item_id": list(range(len(paths))),
                "sid_0": pl.Series(paths, dtype=pl.Int32),
            }
        )

        metrics = calculate_sid_quality(sids, (8,))

        # main exporter: np.quantile([2, 3, 4, 5, 6, 100], .95, nearest)
        self.assertEqual(metrics["collision_percentiles"]["p95"], 100.0)

    def test_cur_total_is_mean_per_level_utilization(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": pl.Series([0, 0, 1, 1], dtype=pl.Int32),
                "sid_1": pl.Series([0, 1, 0, 1], dtype=pl.Int32),
            }
        )

        metrics = calculate_sid_quality(sids, (4, 2))

        quality = metrics["codebook_quality"]
        self.assertEqual(quality["codebook_utilization_rate_at_1_layer"], 0.5)
        self.assertEqual(quality["codebook_utilization_rate_at_2_layer"], 1.0)
        self.assertEqual(quality["codebook_utilization_rate_total"], 0.75)

    def test_no_collisions_have_zero_p95_and_no_similarity(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1],
                "sid_0": pl.Series([0, 1], dtype=pl.Int32),
            }
        )

        metrics = calculate_sid_quality(
            sids,
            (2,),
            sample_embeddings(),
        )

        self.assertEqual(metrics["collision_percentiles"]["p95"], 0.0)
        self.assertNotIn("intra_code_similarity", metrics)

    def test_personalized_quality_is_per_user_catalog_macro(self) -> None:
        frame = pl.DataFrame(
            {
                "uid": [1, 1, 1, 1, 2, 2, 2, 2],
                "item_id": [0, 1, 2, 3, 0, 1, 2, 3],
                "usid_0": pl.Series(
                    [0, 0, 0, 1, 0, 0, 1, 1],
                    dtype=pl.Int32,
                ),
                "usid_1": pl.Series(
                    [0, 0, 1, 0, 0, 0, 1, 1],
                    dtype=pl.Int32,
                ),
            }
        )

        metrics = calculate_user_catalog_sid_quality_from_frame(
            frame,
            (2, 2),
            sample_embeddings(),
        )

        self.assertEqual(metrics["aggregation"], "per_user_catalog_macro")
        self.assertEqual(metrics["user_count"], 2)
        self.assertEqual(metrics["catalog_items_per_user"], 4)
        self.assertAlmostEqual(metrics["independent_code_rate"], 0.625)
        self.assertAlmostEqual(metrics["collision_rate"], 0.375)
        self.assertEqual(metrics["collision_percentiles"]["p95"], 2.0)
        self.assertEqual(metrics["max_collision"], 2.0)
        self.assertEqual(metrics["max_collision_across_users"], 2)

    def test_personalized_quality_rejects_observed_pairs_only(self) -> None:
        incomplete = pl.DataFrame(
            {
                "uid": [1, 1, 2],
                "item_id": [0, 1, 0],
                "usid_0": pl.Series([0, 1, 0], dtype=pl.Int32),
            }
        )

        with self.assertRaisesRegex(ValueError, "full user-catalog grid"):
            calculate_user_catalog_sid_quality_from_frame(incomplete, (2,))

    def test_streaming_personalized_quality_matches_materialized_macro(self) -> None:
        pipeline = FakeUserSidPipeline()
        users = pl.DataFrame({"uid": [1, 2]})
        catalog = pl.DataFrame({"item_id": [0, 1, 2, 3]})

        streamed = calculate_user_catalog_sid_quality(
            pipeline,
            users,
            catalog,
            (2, 2),
            sample_embeddings(),
        )

        self.assertEqual(streamed["user_count"], 2)
        self.assertAlmostEqual(streamed["independent_code_rate"], 0.625)
        self.assertAlmostEqual(streamed["collision_rate"], 0.375)
        self.assertEqual(streamed["collision_percentiles"]["p95"], 2.0)

    def test_fast_streaming_path_matches_reference_semantics(self) -> None:
        pipeline = ShuffledParityUserSidPipeline()
        users = pl.DataFrame({"uid": [10, 20]})
        embeddings = sample_embeddings()
        catalog = embeddings.select("item_id")
        materialized = pl.concat(
            [
                pipeline.update(
                    catalog.with_columns(pl.lit(uid).alias("uid")).select(
                        "uid",
                        "item_id",
                    )
                )
                for uid in users["uid"].to_list()
            ]
        )

        reference = calculate_user_catalog_sid_quality_from_frame(
            materialized,
            (2, 2),
            embeddings,
        )
        streamed = calculate_user_catalog_sid_quality(
            pipeline,
            users,
            catalog,
            (2, 2),
            embeddings,
        )

        self.assertEqual(reference["collision_percentiles"]["p95"], 3.0)
        self.assertEqual(reference["max_collision"], 3.0)
        self.assertIn("prefix_coherence_at_1", reference)
        self.assertIn("prefix_coherence_at_2", reference)
        self.assertEqual(
            reference["intra_code_similarity"],
            reference["prefix_coherence_at_2"],
        )
        assert_nested_metrics_equal(self, streamed, reference)


if __name__ == "__main__":
    unittest.main()
