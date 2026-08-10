from __future__ import annotations

import math
import unittest

import polars as pl

from sids_rank_vs_generate.evaluation.sid_statistics import (
    calculate_sid_statistics,
)


class SidStatisticsTest(unittest.TestCase):
    def test_full_path_occupancy_metrics_and_p95(self) -> None:
        frame = pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 1, 1, 1],
            }
        )

        metrics = calculate_sid_statistics(frame)

        self.assertEqual(metrics["collision_percentiles"]["p95"], 3)
        self.assertEqual(metrics["max_collision"], 3)
        self.assertAlmostEqual(metrics["code_gini"], 0.25)
        expected_entropy = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
        self.assertAlmostEqual(metrics["code_entropy"], expected_entropy)
        self.assertAlmostEqual(
            metrics["code_entropy_normalized"],
            expected_entropy / math.log(2),
        )

    def test_balanced_unique_paths_have_zero_gini_and_unit_normalized_entropy(self) -> None:
        frame = pl.DataFrame(
            {
                "item_id": [0, 1, 2],
                "sid_0": [0, 1, 2],
            }
        )

        metrics = calculate_sid_statistics(frame)

        self.assertEqual(metrics["code_gini"], 0.0)
        self.assertAlmostEqual(metrics["code_entropy"], math.log(3))
        self.assertAlmostEqual(metrics["code_entropy_normalized"], 1.0)


if __name__ == "__main__":
    unittest.main()
