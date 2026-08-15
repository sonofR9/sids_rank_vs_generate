from __future__ import annotations

import unittest
from pathlib import Path

from experiments.diger_recall_overnight import build_lane_stages


class DigerRecallOvernightTest(unittest.TestCase):
    def test_lanes_cover_scale_sid_and_baseline_comparisons(self) -> None:
        both = Path("both.parquet")
        frqud = Path("frqud.parquet")
        lane_zero = build_lane_stages(
            0,
            both_sid_path=both,
            frqud_sid_path=frqud,
        )
        lane_one = build_lane_stages(
            1,
            both_sid_path=both,
            frqud_sid_path=frqud,
        )

        self.assertEqual(len(lane_zero), 4)
        self.assertEqual(len(lane_one), 4)
        self.assertEqual(lane_zero[0].model_variant, "strong128")
        self.assertEqual(lane_one[0].model_variant, "large256")
        self.assertTrue(any(stage.sid_method == "rqkmeans" for stage in lane_zero))
        self.assertTrue(any(stage.sid_variant == "frqud" for stage in lane_one))
        self.assertEqual(
            {stage.protocol for stage in (*lane_zero, *lane_one)},
            {"poster", "three-way"},
        )

    def test_stage_fingerprint_includes_the_sid_source(self) -> None:
        first = build_lane_stages(
            0,
            both_sid_path=Path("a.parquet"),
            frqud_sid_path=Path("f.parquet"),
        )[0]
        second = build_lane_stages(
            0,
            both_sid_path=Path("b.parquet"),
            frqud_sid_path=Path("f.parquet"),
        )[0]

        self.assertNotEqual(first.fingerprint(), second.fingerprint())


if __name__ == "__main__":
    unittest.main()
