from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.evaluation.precomputed_sid_metrics import (
    PrecomputedSidMetricsConfig,
    run_precomputed_sid_metrics,
)
from sids_rank_vs_generate.experiment import main


class PrecomputedSidMetricsTest(unittest.TestCase):
    def _write_inputs(self, directory: Path) -> tuple[Path, Path]:
        sids_dir = directory / "sids"
        sids_dir.mkdir()
        embeddings_path = directory / "embeddings.parquet"
        pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "normalized_embed": [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
            }
        ).write_parquet(embeddings_path)
        pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 0, 1, 1],
                "sid_1": [0, 1, 0, 1],
            }
        ).write_parquet(sids_dir / "unique_paths.parquet")
        pl.DataFrame(
            {
                "item_id": [0, 1, 2, 3],
                "sid_0": [0, 0, 1, 1],
                "sid_1": [0, 0, 1, 1],
            }
        ).write_parquet(sids_dir / "collisions.parquet")
        return sids_dir, embeddings_path

    def test_measures_every_sid_parquet_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sids_dir, embeddings_path = self._write_inputs(directory)

            result = run_precomputed_sid_metrics(
                PrecomputedSidMetricsConfig(
                    sids_dir=sids_dir,
                    embeddings_path=embeddings_path,
                )
            )

            self.assertEqual(result["file_count"], 2)
            unique_metrics = result["files"]["unique_paths.parquet"]["metrics"]
            self.assertIsNone(unique_metrics["path_average_similarity"])
            self.assertAlmostEqual(
                unique_metrics["intra_code_similarity_v2"],
                0.5,
            )
            self.assertEqual(unique_metrics["collision_item_rate"], 0.0)
            self.assertAlmostEqual(
                unique_metrics["pair_refinement_rate_at_1_layer"],
                2 / 3,
            )
            self.assertAlmostEqual(
                unique_metrics["mean_active_branch_fraction_at_2_layer"],
                1.0,
            )
            self.assertAlmostEqual(
                unique_metrics["adjusted_semantic_variance_explained_at_1"],
                1.0,
            )
            self.assertIsNone(
                unique_metrics["adjusted_semantic_variance_explained_at_2"]
            )
            self.assertAlmostEqual(unique_metrics["prefix_coherence_at_1"], 1.0)
            self.assertIsNone(unique_metrics["prefix_coherence_at_2"])
            self.assertAlmostEqual(unique_metrics["code_entropy"], 1.3862943611)
            self.assertEqual(unique_metrics["code_gini"], 0.0)
            self.assertEqual(unique_metrics["p95_collision_group_size"], 0.0)
            self.assertEqual(unique_metrics["cur_total"], 1.0)
            self.assertEqual(unique_metrics["valid_sid_rate"], 1.0)
            collision_metrics = result["files"]["collisions.parquet"]["metrics"]
            self.assertAlmostEqual(collision_metrics["path_average_similarity"], 1.0)
            output_path = Path(result["output"])
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["file_count"],
                2,
            )

    def test_main_supports_metrics_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sids_dir, embeddings_path = self._write_inputs(directory)
            output_path = directory / "batch_metrics.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--sids-metrics-dir",
                        str(sids_dir),
                        "--embeddings-path",
                        str(embeddings_path),
                        "--metrics-output",
                        str(output_path),
                        "--codebook-sizes",
                        "2,2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            self.assertEqual(json.loads(stdout.getvalue())["file_count"], 2)

    def test_rejects_sid_items_without_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            sids_dir, embeddings_path = self._write_inputs(directory)
            embeddings = pl.read_parquet(embeddings_path).filter(pl.col("item_id") != 3)
            embeddings.write_parquet(embeddings_path)

            with self.assertRaisesRegex(ValueError, "1 items without embeddings"):
                run_precomputed_sid_metrics(
                    PrecomputedSidMetricsConfig(
                        sids_dir=sids_dir,
                        embeddings_path=embeddings_path,
                    )
                )


if __name__ == "__main__":
    unittest.main()
