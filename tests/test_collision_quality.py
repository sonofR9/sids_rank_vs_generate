from __future__ import annotations

import unittest

import polars as pl

from sids_rank_vs_generate.evaluation.collision_quality import (
    codebook_quality,
    path_average_similarity,
    prefix_coherence,
    valid_sid_rate,
)


class CollisionQualityTest(unittest.TestCase):
    def test_path_average_similarity_uses_collision_paths(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 0, 1, 1],
                "sid_1": [0, 0, 0, 1],
            }
        )
        embeddings = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "normalized_embed": [
                    [1.0, 0.0],
                    [0.5, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                ],
            }
        )

        self.assertAlmostEqual(path_average_similarity(sids, embeddings), 1.0)

    def test_path_average_similarity_is_none_without_collisions(self) -> None:
        sids = pl.DataFrame({"item_id": [0, 1], "sid_0": [0, 1]})
        embeddings = pl.DataFrame(
            {
                "item_id": [0, 1],
                "normalized_embed": [[1.0, 0.0], [0.0, 1.0]],
            }
        )

        self.assertIsNone(path_average_similarity(sids, embeddings))

    def test_codebook_quality_includes_unused_codes(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 0, 1, 1],
                "sid_1": [0, 1, 0, 1],
            }
        )

        metrics = codebook_quality(sids, (4, 2))

        self.assertEqual(metrics["codebook_utilization_rate_at_1_layer"], 0.5)
        self.assertEqual(metrics["codebook_coverage_at_2_layer"], 0.5)
        self.assertEqual(metrics["load_balance_p25_at_1_layer"], 0.0)
        self.assertAlmostEqual(
            metrics["codebook_utilization_rate_total"],
            4 / 6,
        )

    def test_prefix_coherence_is_macro_average_per_prefix(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 0, 1, 1],
                "sid_1": [0, 1, 0, 1],
            }
        )
        embeddings = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "normalized_embed": [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                ],
            }
        )

        metrics = prefix_coherence(sids, embeddings)

        self.assertAlmostEqual(metrics["prefix_coherence_at_1"], 0.0)
        self.assertIsNone(metrics["prefix_coherence_at_2"])

    def test_valid_sid_rate_checks_nulls_and_codebook_ranges(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": pl.Series([0, 1, 3, None], dtype=pl.Int64),
            }
        )

        self.assertEqual(valid_sid_rate(sids, (2,)), 0.5)


if __name__ == "__main__":
    unittest.main()
