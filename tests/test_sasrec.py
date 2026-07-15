from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from sids_rank_vs_generate.data import PreprocessingConfig, prepare_interactions
from sids_rank_vs_generate.experiments.sasrec import SASRecConfig, main, run_sasrec

from tests.test_preprocessing import sample_interactions


class SASRecSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared = prepare_interactions(
            sample_interactions(),
            PreprocessingConfig(2, 1),
        )
        self.config = SASRecConfig(
            embedding_dim=8,
            max_seq_len=4,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            topk=2,
            seed=7,
            device="cpu",
        )

    def test_short_reproducible_run(self) -> None:
        _, first = run_sasrec(self.prepared, self.config)
        _, second = run_sasrec(self.prepared, self.config)

        self.assertEqual(first.train_losses, second.train_losses)
        self.assertEqual(first.metrics, second.metrics)
        self.assertTrue(all(math.isfinite(loss) for loss in first.train_losses))
        self.assertEqual(first.train_examples, 4)
        self.assertEqual(first.eval_users, 2)
        for value in first.metrics.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_command_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "interactions.parquet"
            metrics_path = Path(tmpdir) / "sasrec.json"
            sample_interactions().write_parquet(parquet_path)
            exit_code = main(
                [
                    "--interactions",
                    str(parquet_path),
                    "--core-min-interactions",
                    "2",
                    "--test-days",
                    "1",
                    "--embedding-dim",
                    "8",
                    "--max-seq-len",
                    "4",
                    "--num-layers",
                    "1",
                    "--num-heads",
                    "2",
                    "--dropout",
                    "0",
                    "--batch-size",
                    "4",
                    "--epochs",
                    "1",
                    "--topk",
                    "2",
                    "--seed",
                    "7",
                    "--device",
                    "cpu",
                    "--metrics-json",
                    str(metrics_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(metrics_path.is_file())


if __name__ == "__main__":
    unittest.main()
