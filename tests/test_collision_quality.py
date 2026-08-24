from __future__ import annotations

import unittest

import polars as pl

from sids_rank_vs_generate.evaluation.collision_quality import (
    adjusted_semantic_variance_explained,
    codebook_quality,
    intra_code_similarity_v2,
    path_average_similarity,
    prefix_coherence,
    sid_distribution_quality,
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

    def test_intra_code_similarity_v2_uses_codes_not_collision_paths(self) -> None:
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
                    [0.0, 1.0],
                ],
            }
        )

        self.assertIsNone(path_average_similarity(sids, embeddings))
        self.assertAlmostEqual(intra_code_similarity_v2(sids, embeddings), 0.5)

    def test_intra_code_similarity_v2_is_none_without_reused_codes(self) -> None:
        sids = pl.DataFrame(
            {
                "item_id": [0, 1],
                "sid_0": [0, 1],
                "sid_1": [1, 0],
            }
        )
        embeddings = pl.DataFrame(
            {
                "item_id": [0, 1],
                "normalized_embed": [[1.0, 0.0], [0.0, 1.0]],
            }
        )

        self.assertIsNone(intra_code_similarity_v2(sids, embeddings))

    def test_prefix_coherence_measures_each_hierarchy_depth(self) -> None:
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
                    [0.0, 1.0],
                ],
            }
        )

        metrics = prefix_coherence(sids, embeddings)

        self.assertAlmostEqual(metrics["prefix_coherence_at_1"], 1.0)
        self.assertIsNone(metrics["prefix_coherence_at_2"])

    def test_distribution_quality_distinguishes_spread_and_collisions(self) -> None:
        spread = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 0, 1, 1],
                "sid_1": [0, 1, 0, 1],
            }
        )
        collapsed = spread.with_columns(
            pl.lit(0).alias("sid_0"),
            pl.lit(0).alias("sid_1"),
        )

        spread_metrics = sid_distribution_quality(spread, (2, 2))
        collapsed_metrics = sid_distribution_quality(collapsed, (2, 2))

        self.assertEqual(spread_metrics["collision_item_rate"], 0.0)
        self.assertEqual(spread_metrics["pair_collision_probability"], 0.0)
        self.assertAlmostEqual(
            spread_metrics["pair_refinement_rate_at_1_layer"],
            2 / 3,
        )
        self.assertAlmostEqual(
            spread_metrics["pair_refinement_rate_at_2_layer"],
            1.0,
        )
        self.assertAlmostEqual(
            spread_metrics["mean_active_branch_fraction_at_2_layer"],
            1.0,
        )
        self.assertAlmostEqual(
            spread_metrics["normalized_conditional_entropy_at_2_layer"],
            1.0,
        )
        self.assertEqual(collapsed_metrics["collision_item_rate"], 1.0)
        self.assertEqual(collapsed_metrics["pair_collision_probability"], 1.0)
        self.assertEqual(collapsed_metrics["pair_refinement_rate_at_1_layer"], 0.0)
        self.assertEqual(collapsed_metrics["pair_refinement_rate_at_2_layer"], 0.0)
        self.assertEqual(
            collapsed_metrics["mean_active_branch_fraction_at_2_layer"],
            0.5,
        )
        self.assertEqual(
            collapsed_metrics["normalized_conditional_entropy_at_2_layer"],
            0.0,
        )

    def test_adjusted_semantic_variance_penalizes_singleton_prefixes(self) -> None:
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
                "normalized_embed": [[0.0], [1.0], [10.0], [11.0]],
            }
        )

        metrics = adjusted_semantic_variance_explained(sids, embeddings)

        self.assertAlmostEqual(
            metrics["adjusted_semantic_variance_explained_at_1"],
            199 / 202,
        )
        self.assertIsNone(
            metrics["adjusted_semantic_variance_explained_at_2"]
        )

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
        self.assertEqual(metrics["codebook_utilization_rate_total"], 0.75)


if __name__ == "__main__":
    unittest.main()
