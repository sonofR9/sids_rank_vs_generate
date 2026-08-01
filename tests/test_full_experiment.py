from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch
from torch import nn

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.evaluation.ranking_stage import (
    RankingConfig,
    TransActRankingAdapter,
)
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    YambdaDatasetConfig,
    run_experiment,
)
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


class FullRQKMeansExperimentTest(unittest.TestCase):
    def test_transact_uses_dcnv2_instead_of_mlp_head(self) -> None:
        config = RankingConfig(
            embedding_dim=4,
            dcn_cross_layers=1,
            dcn_deep_units=(16, 8),
            dcn_low_rank=4,
            transact_d_model=8,
            transact_heads=2,
            transact_ff_dim=16,
        )
        model = TransActRankingAdapter(
            num_users=3,
            sid_codes=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
            sid_cardinalities=(2, 2),
            config=config,
        )

        self.assertIsInstance(model.history_encoder.mlp, nn.Identity)
        expected_input_size = 4 + 2 * 4 + 3 + 8
        self.assertEqual(
            model.ranker.cross_network.layers[0].input_dim,
            expected_input_size,
        )

    def test_rqkmeans_runs_ranking_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            interactions_path = root / "interactions.parquet"
            embeddings_path = root / "embeddings.parquet"

            rows = []
            for uid in range(4):
                for day in range(12):
                    rows.append(
                        {
                            "uid": uid,
                            "item_id": (day + uid) % 6,
                            "timestamp": day * 86_400 + uid,
                        }
                    )
            pl.DataFrame(rows).write_parquet(interactions_path)
            pl.DataFrame(
                {
                    "item_id": list(range(6)),
                    "normalized_embed": [
                        [float(item == dimension) for dimension in range(6)]
                        for item in range(6)
                    ],
                }
            ).write_parquet(embeddings_path)

            config = ExperimentConfig(
                dataset=YambdaDatasetConfig(
                    interactions_path=interactions_path,
                    embeddings_path=embeddings_path,
                    core_min_interactions_per_item=1,
                    test_days=3,
                    timestamp_unit_seconds=1,
                ),
                sid_pipeline=RQKMeansPipeline(
                    RQKMeansConfig(num_levels=2, codebook_size=2, random_state=7)
                ),
                ranking=RankingConfig(
                    models=("dcnv2", "transact"),
                    max_history_len=4,
                    min_history_len=1,
                    batch_size=16,
                    epochs=1,
                    embedding_dim=4,
                    hidden_dim=8,
                    dcn_cross_layers=1,
                    dcn_deep_units=(16, 8),
                    dcn_num_experts=2,
                    dcn_low_rank=4,
                    transact_d_model=8,
                    transact_heads=2,
                    transact_layers=1,
                    transact_ff_dim=16,
                    dropout=0.0,
                    device="cpu",
                ),
                generation=GenerationConfig(
                    max_history_len=4,
                    min_history_len=1,
                    d_model=8,
                    d_ff=16,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    batch_size=16,
                    batch_size_val=16,
                    num_workers=0,
                    max_epochs=1,
                    num_beams=4,
                    recall_k=2,
                    device="cpu",
                ),
                reports_dir=root / "reports",
                seed=7,
            )

            result = run_experiment(config)

            self.assertEqual(set(result["evaluations"]), {"ranking", "generation"})
            self.assertEqual(
                set(result["evaluations"]["ranking"]["models"]),
                {"dcnv2", "transact"},
            )
            self.assertIn("statistics", result["sid"])
            self.assertTrue(Path(result["artifacts"]["semantic_ids"]).is_file())
            self.assertTrue(
                Path(result["evaluations"]["generation"]["checkpoint"]).is_file()
            )


if __name__ == "__main__":
    unittest.main()
