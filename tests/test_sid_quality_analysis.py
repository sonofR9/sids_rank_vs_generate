from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_sid_quality_correlations.py"
POSTER_MODULES = ("matplotlib", "mlflow", "pandas", "scipy")
POSTER_AVAILABLE = all(importlib.util.find_spec(name) for name in POSTER_MODULES)


@unittest.skipUnless(POSTER_AVAILABLE, "poster analysis extras are not installed")
class SidQualityAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("sid_quality_analysis", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_detects_generation_recall_at_100(self) -> None:
        pd = self.module.pd
        runs = pd.DataFrame(
            {
                "metrics.validation.recall_at_100": [0.4, 0.5],
                "metrics.evaluations.generation.recall_at_100": [0.1, 0.2],
                "metrics.evaluations.generation.recall_at_50": [0.3, 0.4],
            }
        )
        self.assertEqual(
            self.module._detect_recall_metric(runs),
            "metrics.evaluations.generation.recall_at_100",
        )

    def test_join_uses_the_complete_configuration(self) -> None:
        pd = self.module.pd
        configuration = {
            "model": "rqvae",
            "init": "rqkmeans",
            "assignment": "nearest",
            "collision_resolution": "none",
            "codebook_size": 256,
            "num_levels": 3,
            "seed": 42,
            "dataset": "50m",
            "data_scope": "full",
        }
        quality = pd.DataFrame(
            [
                {
                    **configuration,
                    "run_id": "quality",
                    "run_name": "sid-quality-rqvae",
                    "experiment_name": "quality",
                    "experiment_id": "12",
                    "start_time": 2,
                    "git_commit": "abc",
                    "git_branch": "branch",
                    "metrics.sid.independent_code_rate": 0.9,
                }
            ]
        )
        recall = pd.DataFrame(
            [
                {
                    **configuration,
                    "run_id": "recall",
                    "run_name": "poster-rqvae",
                    "experiment_name": "recall",
                    "experiment_id": "1",
                    "start_time": 1,
                    "git_commit": "def",
                    "git_branch": "poster",
                    "metrics.evaluations.generation.recall_at_100": 0.11,
                },
                {
                    **configuration,
                    "num_levels": 4,
                    "run_id": "other",
                    "run_name": "poster-rqvae-l4",
                    "experiment_name": "recall",
                    "experiment_id": "1",
                    "start_time": 1,
                    "git_commit": "def",
                    "git_branch": "poster",
                    "metrics.evaluations.generation.recall_at_100": 0.12,
                },
            ]
        )
        merged, missing_quality, missing_recall = self.module._join_runs(
            quality,
            recall,
            "metrics.evaluations.generation.recall_at_100",
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(float(merged.iloc[0]["recall_100"]), 0.11)
        self.assertEqual(merged.iloc[0]["method"], "RQVAE")
        self.assertTrue(missing_quality.empty)
        self.assertEqual(len(missing_recall), 1)

    def test_constant_metric_has_undefined_correlations(self) -> None:
        pd = self.module.pd
        frame = pd.DataFrame(
            {
                "metric": [1.0, 1.0, 1.0],
                "recall_100": [0.08, 0.10, 0.12],
            }
        )
        n, pearson, spearman = self.module._correlations(frame, "metric")
        self.assertEqual(n, 3)
        self.assertTrue(math.isnan(pearson))
        self.assertTrue(math.isnan(spearman))


if __name__ == "__main__":
    unittest.main()
