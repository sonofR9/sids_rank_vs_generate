from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from sids_rank_vs_generate.models.ranker_pair_encoder import (
    RankerPairEncoder,
    build_pair_features,
)
from sids_rank_vs_generate.training.ranker_pair_encoder import (
    RankerPairEncoderTrainingConfig,
    encode_ranker_pairs,
    encode_ranker_pairs_indexed,
    fit_ranker_pair_encoder,
    fit_ranker_pair_encoder_dense,
)


class RankerPairEncoderTest(unittest.TestCase):
    @staticmethod
    def _fit_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        items = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
        profiles = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]
        )
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return items, profiles, targets

    def test_pair_features_match_the_declared_formula(self) -> None:
        items = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        profiles = torch.tensor([[4.0, 3.0], [2.0, 0.0]])

        actual = build_pair_features(items, profiles)

        normalized_items = F.normalize(items, dim=-1)
        normalized_profiles = F.normalize(profiles, dim=-1)
        expected = F.normalize(
            torch.cat(
                (
                    normalized_items,
                    normalized_profiles,
                    normalized_items * normalized_profiles,
                    torch.abs(normalized_items - normalized_profiles),
                ),
                dim=-1,
            ),
            dim=-1,
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_forward_returns_logits_and_unit_norm_embeddings(self) -> None:
        model = RankerPairEncoder(
            embedding_dim=3,
            hidden_dims=(8,),
            representation_dim=5,
            dropout=0.0,
        )
        output = model(torch.randn(4, 3), torch.randn(4, 3))

        self.assertEqual(output.logits.shape, (4,))
        self.assertEqual(output.embeddings.shape, (4, 5))
        self.assertTrue(
            torch.allclose(
                output.embeddings.norm(dim=-1),
                torch.ones(4),
                atol=1e-6,
            )
        )

    def test_training_is_deterministic_and_reports_fit_metrics(self) -> None:
        items, profiles, targets = self._fit_tensors()
        config = RankerPairEncoderTrainingConfig(
            batch_size=4,
            epochs=3,
            learning_rate=0.02,
            pos_weight="auto",
            device="cpu",
            seed=17,
            show_progress=False,
        )
        first = RankerPairEncoder(
            2,
            hidden_dims=(8,),
            representation_dim=4,
            dropout=0.1,
        )
        second = RankerPairEncoder(
            2,
            hidden_dims=(8,),
            representation_dim=4,
            dropout=0.1,
        )
        rows = torch.arange(items.shape[0])

        first_result = fit_ranker_pair_encoder(
            first,
            items,
            profiles,
            rows,
            rows,
            targets,
            config,
        )
        second_result = fit_ranker_pair_encoder(
            second,
            items,
            profiles,
            rows,
            rows,
            targets,
            config,
        )

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result.examples, 8)
        self.assertEqual(first_result.positives, 2)
        self.assertEqual(first_result.negatives, 6)
        self.assertEqual(first_result.pos_weight, 3.0)
        self.assertEqual(first_result.table_device, "cpu")
        self.assertEqual(len(first_result.epoch_losses), 3)
        self.assertTrue(all(math.isfinite(loss) for loss in first_result.epoch_losses))
        self.assertIn("roc_auc", first_result.metrics)
        self.assertIn("pr_auc", first_result.metrics)
        self.assertIn("balanced_accuracy", first_result.metrics)
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))

    def test_batched_encoding_is_ordered_and_has_no_gradients(self) -> None:
        items, profiles, _ = self._fit_tensors()
        model = RankerPairEncoder(
            2,
            hidden_dims=(8,),
            representation_dim=4,
            dropout=0.5,
        )
        model.train()

        first = encode_ranker_pairs(
            model,
            items,
            profiles,
            batch_size=3,
            device="cpu",
        )
        second = encode_ranker_pairs(
            model,
            items,
            profiles,
            batch_size=5,
            device="cpu",
        )

        self.assertTrue(model.training)
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.allclose(first, second, atol=1e-7, rtol=1e-6))
        self.assertTrue(
            torch.allclose(first.norm(dim=-1), torch.ones(8), atol=1e-6)
        )

    def test_indexed_encoding_gathers_tables_in_requested_order(self) -> None:
        item_table = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        profile_table = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        item_rows = torch.tensor([1, 0, 1, 1, 0])
        profile_rows = torch.tensor([0, 1, 1, 0, 0])
        model = RankerPairEncoder(
            2,
            hidden_dims=(8,),
            representation_dim=4,
            dropout=0.0,
        )

        indexed = encode_ranker_pairs_indexed(
            model,
            item_table,
            profile_table,
            item_rows,
            profile_rows,
            batch_size=2,
            device="cpu",
        )
        dense = encode_ranker_pairs(
            model,
            item_table[item_rows],
            profile_table[profile_rows],
            batch_size=3,
            device="cpu",
        )

        self.assertFalse(indexed.requires_grad)
        self.assertTrue(torch.allclose(indexed, dense, atol=1e-7, rtol=1e-6))

    def test_fp16_requires_cuda(self) -> None:
        items, profiles, targets = self._fit_tensors()
        with self.assertRaisesRegex(ValueError, "only on CUDA"):
            fit_ranker_pair_encoder_dense(
                RankerPairEncoder(2, hidden_dims=(4,), representation_dim=2),
                items,
                profiles,
                targets,
                RankerPairEncoderTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    precision="fp16",
                    device="cpu",
                    show_progress=False,
                ),
            )

    def test_fit_rejects_a_single_class(self) -> None:
        items = torch.randn(4, 2)
        profiles = torch.randn(4, 2)
        with self.assertRaisesRegex(ValueError, "both classes"):
            fit_ranker_pair_encoder_dense(
                RankerPairEncoder(2, hidden_dims=(4,), representation_dim=2),
                items,
                profiles,
                torch.ones(4),
                RankerPairEncoderTrainingConfig(
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    show_progress=False,
                ),
            )


if __name__ == "__main__":
    unittest.main()
