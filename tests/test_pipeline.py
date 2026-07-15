from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sids_rank_vs_generate.pipeline import PipelineConfig, run_pipeline

from tests.test_preprocessing import sample_interactions


ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_pipeline_and_run_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "interactions.parquet"
            summary_path = Path(tmpdir) / "summary.json"
            sample_interactions().write_parquet(parquet_path)

            prepared = run_pipeline(
                PipelineConfig(
                    interactions_path=parquet_path,
                    core_min_interactions_per_item=2,
                    test_days=1,
                    seed=7,
                )
            )
            self.assertEqual(prepared.catalog_size, 3)

            command = [
                sys.executable,
                str(ROOT / "run.py"),
                "--interactions",
                str(parquet_path),
                "--core-min-interactions",
                "2",
                "--test-days",
                "1",
                "--summary-json",
                str(summary_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["catalog_size"], 3)
            self.assertEqual(json.loads(summary_path.read_text()), summary)


if __name__ == "__main__":
    unittest.main()
