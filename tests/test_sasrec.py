from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import polars as pl
import torch

from sids_rank_vs_generate.data import PreprocessingConfig, prepare_interactions
from sids_rank_vs_generate.experiments.sasrec import (
    SASRecConfig,
    SASRecModel,
    _apply_negative_logq_correction,
    _build_in_batch_item_counts,
    _cuda_autocast_dtype,
    _sample_in_batch_negative_item_ids,
    main,
    run_sasrec,
)

from tests.test_preprocessing import sample_interactions


DAY = 24 * 60 * 60


def in_batch_interactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1, 2, 1, 2, 1, 2, 1, 2],
            "item_id": [10, 50, 20, 60, 30, 70, 40, 80],
            "timestamp": [
                0,
                1,
                DAY,
                DAY + 1,
                2 * DAY,
                2 * DAY + 1,
                4 * DAY,
                4 * DAY + 1,
            ],
        }
    )


class SASRecSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prepared = prepare_interactions(
            sample_interactions(),
            PreprocessingConfig(2, 1),
        )
        self.in_batch_prepared = prepare_interactions(
            in_batch_interactions(),
            PreprocessingConfig(1, 1),
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

    def test_in_batch_logq_run_is_reproducible(self) -> None:
        config = replace(
            self.config,
            training_objective="in-batch-logq",
            training_negatives=3,
        )
        _, first = run_sasrec(self.in_batch_prepared, config)
        _, second = run_sasrec(self.in_batch_prepared, config)

        self.assertEqual(first.train_losses, second.train_losses)
        self.assertEqual(first.metrics, second.metrics)
        self.assertTrue(all(math.isfinite(loss) for loss in first.train_losses))

    def test_in_batch_run_without_logq_is_reproducible(self) -> None:
        config = replace(
            self.config,
            training_objective="in-batch",
            training_negatives=3,
        )
        _, first = run_sasrec(self.in_batch_prepared, config)
        _, second = run_sasrec(self.in_batch_prepared, config)

        self.assertEqual(first.train_losses, second.train_losses)
        self.assertEqual(first.metrics, second.metrics)
        self.assertTrue(all(math.isfinite(loss) for loss in first.train_losses))

    def test_history_concat_uses_frozen_pretrained_features(self) -> None:
        content = torch.arange(
            self.prepared.catalog_size * 3,
            dtype=torch.float32,
        ).reshape(self.prepared.catalog_size, 3)
        config = replace(
            self.config,
            pretrained_embedding_mode="history-concat",
        )

        model, result = run_sasrec(
            self.prepared,
            config,
            initial_item_embeddings=content,
        )

        self.assertFalse(result.content_initialized)
        self.assertTrue(result.content_features_used)
        self.assertIsNotNone(model.pretrained_item_embeddings)
        self.assertIsNotNone(model.history_input_projection)
        self.assertFalse(model.pretrained_item_embeddings.weight.requires_grad)
        self.assertEqual(
            model.history_input_projection.in_features,
            self.config.embedding_dim + 3,
        )
        self.assertEqual(
            model.history_input_projection.out_features,
            self.config.embedding_dim,
        )

    def test_history_concat_accepts_content_width_different_from_model(self) -> None:
        content = torch.ones((self.prepared.catalog_size, 5))
        config = replace(
            self.config,
            pretrained_embedding_mode="history-concat",
        )
        model = SASRecModel(
            self.prepared.catalog_size,
            config,
            initial_item_embeddings=content,
        )

        history = torch.tensor([[0, 1]], dtype=torch.long)
        encoded = model.encode(history, torch.tensor([2]))
        self.assertEqual(tuple(encoded.shape), (1, self.config.embedding_dim))
        self.assertTrue(torch.isfinite(encoded).all())

    def test_history_concat_requires_pretrained_embeddings(self) -> None:
        config = replace(
            self.config,
            pretrained_embedding_mode="history-concat",
        )
        with self.assertRaisesRegex(ValueError, "requires pretrained"):
            SASRecModel(self.prepared.catalog_size, config)

    def test_in_batch_sampling_uses_histories_and_targets(self) -> None:
        padding_item_id = 10
        histories = torch.tensor(
            [
                [0, 1, padding_item_id],
                [0, 8, 9],
            ],
            dtype=torch.long,
        )
        targets = torch.tensor([2, 5], dtype=torch.long)
        sampled = _sample_in_batch_negative_item_ids(
            histories,
            targets,
            torch.tensor([0, 1], dtype=torch.long),
            padding_item_id=padding_item_id,
            count=100,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertEqual(tuple(sampled.shape), (2, 100))
        self.assertEqual(set(sampled[0].tolist()), {5, 8, 9})
        self.assertEqual(set(sampled[1].tolist()), {1, 2})
        self.assertFalse(sampled.eq(targets.unsqueeze(1)).any())
        for row, history in enumerate(histories):
            history_items = history[history.ne(padding_item_id)]
            self.assertFalse(
                sampled[row].unsqueeze(1).eq(history_items).any()
            )

    def test_in_batch_sampling_rejects_single_item_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "another user's"):
            _sample_in_batch_negative_item_ids(
                torch.tensor([[4, 4], [4, 4]], dtype=torch.long),
                torch.tensor([4, 4], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
                padding_item_id=10,
                count=2,
                generator=torch.Generator().manual_seed(11),
            )

    def test_in_batch_sampling_rejects_batch_with_one_user(self) -> None:
        with self.assertRaisesRegex(ValueError, "another user's"):
            _sample_in_batch_negative_item_ids(
                torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                torch.tensor([2, 3], dtype=torch.long),
                torch.tensor([0, 0], dtype=torch.long),
                padding_item_id=4,
                count=2,
                generator=torch.Generator().manual_seed(11),
            )

    def test_in_batch_counts_match_all_history_and_target_pools(self) -> None:
        counts = _build_in_batch_item_counts(
            {
                "first": [0, 1, 1, 2],
                "second": [2, 0],
                "too_short": [1],
            },
            num_items=3,
            max_seq_len=2,
        )

        torch.testing.assert_close(
            counts,
            torch.tensor([3.0, 5.0, 2.0]),
        )

    def test_logq_correction_changes_only_negative_logits(self) -> None:
        logits = torch.zeros((2, 3), dtype=torch.float32)
        negative_ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
        q_counts = torch.tensor([1.0, 2.0, 4.0])
        item_log_q = torch.log(q_counts / q_counts.sum())

        corrected = _apply_negative_logq_correction(
            logits,
            negative_ids,
            item_log_q,
        )
        expected_negative_logits = torch.log(torch.tensor(7.0)) - torch.log(
            q_counts[negative_ids]
        )

        torch.testing.assert_close(corrected[:, 0], logits[:, 0])
        torch.testing.assert_close(corrected[:, 1:], expected_negative_logits)

    def test_v100_uses_float16_autocast(self) -> None:
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(
                _cuda_autocast_dtype(torch.device("cuda")),
                torch.float16,
            )

    def test_command_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "interactions.parquet"
            embeddings_path = Path(tmpdir) / "embeddings.parquet"
            metrics_path = Path(tmpdir) / "sasrec.json"
            in_batch_interactions().write_parquet(parquet_path)
            pl.DataFrame(
                {
                    "item_id": [10, 20, 30, 40, 50, 60, 70, 80],
                    "normalized_embed": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ).write_parquet(embeddings_path)
            exit_code = main(
                [
                    "--interactions",
                    str(parquet_path),
                    "--embeddings",
                    str(embeddings_path),
                    "--core-min-interactions",
                    "1",
                    "--test-days",
                    "1",
                    "--timestamp-unit-seconds",
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
                    "--training-objective",
                    "in-batch-logq",
                    "--training-negatives",
                    "3",
                    "--pretrained-embedding-mode",
                    "history-concat",
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
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["config"]["training_objective"],
                "in-batch-logq",
            )
            self.assertEqual(
                payload["config"]["pretrained_embedding_mode"],
                "history-concat",
            )
            self.assertTrue(payload["content_features_used"])


if __name__ == "__main__":
    unittest.main()
