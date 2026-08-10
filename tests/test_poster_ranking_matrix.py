from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from sids_rank_vs_generate.experiment import load_experiment_config


SCRIPT = Path("scripts/run_poster_ranking_matrix.sh")
CONFIG = Path("experiments/poster_sid_config.py")


class PosterRankingMatrixTest(unittest.TestCase):
    def test_full_matrix_has_28_unique_none_ranking_runs(self) -> None:
        environment = {
            **os.environ,
            "PYTHON_BIN": "/bin/echo",
            "RUN_SET": "ranking_none_all",
            "GPU": "0",
        }
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        starts = [
            line for line in result.stdout.splitlines()
            if line.startswith("Starting poster-ranking-")
        ]
        calls = [
            line for line in result.stdout.splitlines()
            if line.startswith("run.py ")
        ]
        self.assertEqual(len(starts), 28)
        self.assertEqual(len(set(starts)), 28)
        self.assertEqual(len(calls), 28)
        self.assertTrue(all("-none-" in line for line in starts))
        self.assertTrue(all("--skip-generation" in line for line in calls))
        self.assertTrue(all("--skip-ranking" not in line for line in calls))
        self.assertTrue(all("token" not in line for line in starts))

    def test_config_tags_ranking_scope_and_uses_accelerated_schedule(self) -> None:
        environment = {
            "SID_MODEL": "rqkmeans",
            "R3_INIT": "rqkmeans",
            "ASSIGNMENT_METHOD": "nearest",
            "COLLISION_RESOLUTION": "none",
            "EVALUATION_SCOPE": "ranking",
            "NUM_LEVELS": "3",
            "CODEBOOK_SIZE": "256",
        }
        with patch.dict(os.environ, environment, clear=True):
            experiment = load_experiment_config(str(CONFIG))

        tags = dict(experiment.tracking.tags)
        ranking = experiment.training.ranking
        self.assertEqual(tags["evaluation_scope"], "ranking")
        self.assertEqual(ranking.dcnv2_training.batch_size, 32_768)
        self.assertEqual(ranking.dcnv2_training.epochs, 2)
        self.assertEqual(ranking.transact_training.batch_size, 8_192)
        self.assertEqual(ranking.transact_training.epochs, 1)


if __name__ == "__main__":
    unittest.main()
