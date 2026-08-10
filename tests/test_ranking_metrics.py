from __future__ import annotations

import math
import unittest

import polars as pl
import torch

from sids_rank_vs_generate.data.ranking import (
    DENSE_COLUMNS,
    collate_ranking_frame,
)
from sids_rank_vs_generate.evaluation.ranking_metrics import ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_pairwise_accuracy_reports_eligible_comparison_count(self) -> None:
        metrics = ranking_metrics(
            targets=torch.tensor([0.0, 1.0, 0.0]),
            probabilities=torch.tensor([0.1, 0.9, 0.2]),
            users=torch.tensor([7, 7, 7]),
            timestamps=torch.tensor([0.0, 600.0, 1_501.0]),
        )

        self.assertEqual(metrics["pairwise_comparisons"], 1)
        self.assertEqual(metrics["pairwise_users"], 1)
        self.assertEqual(metrics["pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["pairwise_accuracy_ci95_low"], 1.0)
        self.assertEqual(metrics["pairwise_accuracy_ci95_high"], 1.0)

    def test_columnar_collation_converts_timestamp_bins_to_seconds(self) -> None:
        row = {
            "timestamp": 100,
            "history_timestamps": [98, 99],
            "history_item_idxs": [1, 2],
            "uid_hash": 3,
            "sid_0": 4,
            "sid_0_hash": 5,
            "artist_ids": [10],
            "artist_ids_hash_0": [11],
            "artist_ids_hash_1": [12],
            "album_ids": [20],
            "album_ids_hash_0": [21],
            "album_ids_hash_1": [22],
            "user_idx": 0,
            "history_item_idx": 2,
            "track_length_seconds": 180,
            "target": 1.0,
            **{column: 1.0 for column in DENSE_COLUMNS},
        }

        batch = collate_ranking_frame(
            pl.DataFrame([row]),
            [0],
            timestamp_unit_seconds=5.0,
        )

        self.assertEqual(batch["candidate_timestamp"].item(), 500.0)
        self.assertTrue(
            torch.allclose(
                batch["history_time_delta_z_flat"],
                torch.tensor([math.log1p(10), math.log1p(5)]),
            )
        )


if __name__ == "__main__":
    unittest.main()
