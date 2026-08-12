from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import torch

from sids_rank_vs_generate.data.ranking import (
    DENSE_COLUMNS,
    collate_ranking_frame,
)
from sids_rank_vs_generate.evaluation.ranking_metrics import ranking_metrics
from sids_rank_vs_generate.evaluation.ranking_stage import RankingConfig
from sids_rank_vs_generate.models.ranking import TransActDcnV2Ranker
from sids_rank_vs_generate.models.transact import CandidateTokenTransformer
from sids_rank_vs_generate.training.ranking import train_ranking_model


class _ConstantRanker(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return self.logit.expand_as(batch["target"])


class _UnusedLoader:
    def __iter__(self):
        raise AssertionError("test data must not be inspected while tuning")


class RankingMetricsTest(unittest.TestCase):
    def test_default_ranking_schedule_matches_tuned_window(self) -> None:
        config = RankingConfig()

        self.assertFalse(config.hyperparameter_tuning)
        self.assertEqual(config.train_days, 270.0)
        self.assertEqual(config.validation_days, 7.0)
        self.assertEqual(config.dcnv2_training.batch_size, 32_768)
        self.assertEqual(config.dcnv2_training.epochs, 2)
        self.assertEqual(config.dcnv2_training.learning_rate, 2e-3)
        self.assertEqual(config.transact_training.batch_size, 8_192)
        self.assertEqual(config.transact_training.epochs, 1)
        self.assertEqual(config.transact_training.learning_rate, 5e-4)
        self.assertEqual(config.enabled_models, ("dcnv2", "transact"))

    def test_ranking_can_enable_only_transact(self) -> None:
        config = RankingConfig(enabled_models=("transact",))

        self.assertEqual(config.enabled_models, ("transact",))
        with self.assertRaisesRegex(ValueError, "unknown ranking models"):
            RankingConfig(enabled_models=("other",))

    def test_pairwise_accuracy_reports_eligible_comparison_count(self) -> None:
        metrics = ranking_metrics(
            targets=torch.tensor([0.0, 1.0, 0.0]),
            probabilities=torch.tensor([0.1, 0.9, 0.2]),
            users=torch.tensor([7, 7, 7]),
            timestamps=torch.tensor([0.0, 600.0, 1_501.0]),
        )

        self.assertEqual(metrics["pairwise_comparisons"], 1)
        self.assertEqual(metrics["pairwise_users"], 1)
        self.assertEqual(metrics["pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["pairwise_accuracy_ci95_low"], 1.0)
        self.assertEqual(metrics["pairwise_accuracy_ci95_high"], 1.0)

    def test_columnar_collation_converts_timestamp_bins_to_seconds(self) -> None:
        row = {
            "timestamp": 100,
            "history_timestamps": [98, 99],
            "history_item_idxs": [1, 2],
            "history_sid_0": [4, 6],
            "uid_hash": 3,
            "sid_0": 4,
            "sid_0_hash": 5,
            "artist_ids": [10],
            "artist_ids_hash_0": [11],
            "artist_ids_hash_1": [12],
            "album_ids": [20],
            "album_ids_hash_0": [21],
            "album_ids_hash_1": [22],
            "user_idx": 0,
            "history_item_idx": 2,
            "track_length_seconds": 180,
            "target": 1.0,
            **{column: 1.0 for column in DENSE_COLUMNS},
        }

        batch = collate_ranking_frame(
            pl.DataFrame([row]),
            [0],
            timestamp_unit_seconds=5.0,
        )

        self.assertEqual(batch["candidate_timestamp"].item(), 500.0)
        self.assertTrue(
            torch.allclose(
                batch["history_time_delta_z_flat"],
                torch.tensor([math.log1p(10), math.log1p(5)]),
            )
        )
        self.assertEqual(batch["history_sid_values_flat"].tolist(), [4, 6])
        self.assertEqual(batch["history_sid_levels_flat"].tolist(), [0, 0])
        self.assertEqual(batch["history_lengths"].tolist(), [2])

    def test_sid_history_expands_levels_into_separate_tokens(self) -> None:
        row = {
            "timestamp": 100,
            "history_timestamps": [98, 99],
            "history_item_idxs": [1, 2],
            "history_sid_0": [4, 6],
            "history_sid_1": [5, 7],
            "uid_hash": 3,
            "sid_0": 4,
            "sid_0_hash": 5,
            "sid_1": 5,
            "sid_1_hash": 6,
            "artist_ids": [10],
            "artist_ids_hash_0": [11],
            "artist_ids_hash_1": [12],
            "album_ids": [20],
            "album_ids_hash_0": [21],
            "album_ids_hash_1": [22],
            "user_idx": 0,
            "history_item_idx": 2,
            "track_length_seconds": 180,
            "target": 1.0,
            **{column: 1.0 for column in DENSE_COLUMNS},
        }

        batch = collate_ranking_frame(pl.DataFrame([row]), [0])

        self.assertEqual(
            batch["history_sid_values_flat"].tolist(), [4, 5, 6, 7]
        )
        self.assertEqual(
            batch["history_sid_levels_flat"].tolist(), [0, 1, 0, 1]
        )
        self.assertEqual(batch["history_lengths"].tolist(), [4])
        self.assertTrue(
            torch.allclose(
                batch["history_time_delta_z_flat"],
                torch.tensor(
                    [math.log1p(2), math.log1p(2), math.log1p(1), math.log1p(1)]
                ),
            )
        )
        self.assertEqual(set(batch["sparse_features"]), {"uid", "sid_0", "sid_1"})

    def test_sid_levels_have_independent_history_embedding_tables(self) -> None:
        model = CandidateTokenTransformer(
            num_users=1,
            num_items=2,
            num_event_types=3,
            item_emb_dim=3,
            d_model=4,
            n_heads=1,
            sid_cardinalities=(5, 7),
        )
        with torch.no_grad():
            model.sid_embeddings[0].weight.fill_(1.0)
            model.sid_embeddings[1].weight.fill_(2.0)

        vectors = model.embed_history_sids(
            torch.tensor([3, 3]),
            torch.tensor([0, 1]),
        )

        self.assertTrue(torch.equal(vectors[0], torch.ones(3)))
        self.assertTrue(torch.equal(vectors[1], torch.full((3,), 2.0)))

    def test_transact_dcn_head_keeps_candidate_sid_features(self) -> None:
        data = SimpleNamespace(
            num_users=2,
            num_history_items=4,
            sid_columns=("sid_0", "sid_1"),
            sid_cardinalities=(3, 4),
            dense_train=pl.DataFrame(
                {column: [0.0, 1.0, 2.0] for column in DENSE_COLUMNS}
            ),
        )
        config = RankingConfig(
            embedding_dim=4,
            dense_bins=2,
            dcn_deep_units=(8, 4),
            dcn_num_experts=2,
            dcn_low_rank=2,
            transact_d_model=4,
            transact_heads=1,
            transact_ff_dim=8,
        )

        model = TransActDcnV2Ranker(data, config)

        self.assertEqual(
            model.ranker.sparse_feature_names,
            ("uid", "sid_0", "sid_1"),
        )

    def test_tuning_evaluates_validation_without_inspecting_test(self) -> None:
        batch = {
            "target": torch.tensor([0.0, 1.0]),
            "user_idx": torch.tensor([0, 0]),
            "candidate_timestamp": torch.tensor([0.0, 1.0]),
        }
        config = SimpleNamespace(
            device="cpu",
            learning_rate=1e-3,
            weight_decay=0.0,
            epochs=1,
            grad_clip=1.0,
            show_progress=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = train_ranking_model(
                _ConstantRanker(),
                [batch],
                [batch],
                _UnusedLoader(),
                config,
                Path(directory) / "model.pt",
            )

        self.assertEqual(result["mode"], "hyperparameter_tuning")
        self.assertEqual(result["checkpoint_metric"], "roc_auc")
        self.assertEqual(result["best_epoch"], 1)
        self.assertIsNotNone(result["best_validation"])
        self.assertIsNone(result["test"])


if __name__ == "__main__":
    unittest.main()
