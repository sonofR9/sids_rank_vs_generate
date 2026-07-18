from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

from sids_rank_vs_generate.pipeline import PipelineConfig, run_pipeline

from tests.test_preprocessing import sample_interactions


class PipelineTest(unittest.TestCase):
    def test_pipeline_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "interactions.parquet"
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

    def test_pipeline_seed_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "interactions.parquet"
            sample_interactions().write_parquet(parquet_path)
            config = PipelineConfig(
                interactions_path=parquet_path,
                core_min_interactions_per_item=2,
                test_days=1,
                seed=17,
            )

            run_pipeline(config)
            first = (random.random(), np.random.rand(), torch.rand(1).item())
            run_pipeline(config)
            second = (random.random(), np.random.rand(), torch.rand(1).item())
            self.assertEqual(first, second)

    def test_pipeline_can_restrict_the_eligible_item_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "interactions.parquet"
            sample_interactions().write_parquet(parquet_path)
            prepared = run_pipeline(
                PipelineConfig(
                    interactions_path=parquet_path,
                    core_min_interactions_per_item=2,
                    test_days=1,
                    seed=7,
                ),
                eligible_items=pl.DataFrame({"item_id": [10, 20]}),
            )

            self.assertEqual(prepared.catalog_size, 2)
            self.assertEqual(prepared.data.height, 6)


if __name__ == "__main__":
    unittest.main()
