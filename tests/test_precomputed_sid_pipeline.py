from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.datasets import DatasetFiles
from sids_rank_vs_generate.sid_experiment import SidExperimentConfig, run_sid_experiment
from sids_rank_vs_generate.sids import PrecomputedSidConfig, PrecomputedSidPipeline
from tests.test_preprocessing import sample_interactions


class PrecomputedSidPipelineTest(unittest.TestCase):
    def test_remaps_original_ids_through_the_common_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            interactions_path = directory / "interactions.parquet"
            embeddings_path = directory / "embeddings.parquet"
            sids_path = directory / "sids.parquet"

            pl.concat([sample_interactions()] * 3).write_parquet(interactions_path)
            pl.DataFrame(
                {
                    "item_id": [30, 20, 10, 999],
                    "embed": [
                        [3.0, 0.0],
                        [2.0, 0.0],
                        [1.0, 0.0],
                        [9.0, 9.0],
                    ],
                }
            ).write_parquet(embeddings_path)
            pl.DataFrame(
                {
                    "item_id": [30, 10, 20],
                    "catalog_item_idx": [99, 98, 97],
                    "sid_0": [2, 0, 1],
                    "sid_1": [0, 0, 0],
                }
            ).write_parquet(sids_path)

            result = run_sid_experiment(
                SidExperimentConfig(
                    dataset=DatasetFiles(
                        size="50m",
                        interactions_path=interactions_path,
                        embeddings_path=embeddings_path,
                    ),
                    sid_pipeline=PrecomputedSidPipeline(
                        PrecomputedSidConfig(
                            path=sids_path,
                            codebook_sizes=(3, 1),
                            expected_catalog_size=3,
                        )
                    ),
                    reports_dir=directory / "reports",
                    core_min_interactions_per_item=2,
                    test_days=1,
                )
            )

            mapping = pl.read_parquet(result["artifacts"]["item_sid_mapping"])
            self.assertEqual(mapping["item_id"].to_list(), [10, 20, 30])
            self.assertEqual(mapping["catalog_item_idx"].to_list(), [0, 1, 2])
            self.assertEqual(mapping["sid_0"].to_list(), [0, 1, 2])

    def test_rejects_an_incomplete_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sids.parquet"
            pl.DataFrame(
                {"item_id": [10], "sid_0": pl.Series([0], dtype=pl.Int32)}
            ).write_parquet(path)
            pipeline = PrecomputedSidPipeline(
                PrecomputedSidConfig(path=path, codebook_sizes=(2,))
            )
            pipeline.set_item_mapping(
                pl.DataFrame({"original_item_id": [10, 20], "item_id": [0, 1]})
            )
            with self.assertRaisesRegex(ValueError, "does not cover"):
                pipeline.train(pl.DataFrame({"item_id": [0, 1]}))


if __name__ == "__main__":
    unittest.main()
