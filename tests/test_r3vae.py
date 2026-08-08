from __future__ import annotations

import math
import unittest

import polars as pl
import torch

from sids_rank_vs_generate.models.r3vae import (
    R3VAE,
    preference_discrimination_loss,
)
from sids_rank_vs_generate.sids.r3vae import R3VAEConfig, R3VAESid


class R3VAEModelTest(unittest.TestCase):
    def test_reference_projection_leaves_orthogonal_residual(self) -> None:
        model = R3VAE(3, (4, 4))
        with torch.no_grad():
            model.reference_vector.copy_(torch.tensor([[2.0, 0.0, 0.0]]))
        embeddings = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.0]])

        projection, residual = model.reference_projection(embeddings)

        torch.testing.assert_close(projection[:, 0], embeddings[:, 0])
        torch.testing.assert_close(residual[:, 0], torch.zeros(2))
        torch.testing.assert_close(
            residual @ torch.tensor([1.0, 0.0, 0.0]),
            torch.zeros(2),
        )

    def test_forward_is_finite_and_propagates_gradients(self) -> None:
        torch.manual_seed(7)
        model = R3VAE(4, (3, 3))
        output = model(torch.randn(6, 4))

        self.assertEqual(tuple(output.indices.shape), (6, 2))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.reference_vector.grad)
        self.assertTrue(torch.isfinite(model.reference_vector.grad).all())
        for codebook in model.quantizer.codebooks:
            self.assertIsNotNone(codebook.grad)
            self.assertTrue(torch.isfinite(codebook.grad).all())

    def test_pd_matches_equation_for_orthogonal_codes(self) -> None:
        codebook = torch.eye(3)
        loss = preference_discrimination_loss(codebook, temperature=2.0)
        self.assertAlmostEqual(loss.item(), -2.0, places=6)


class R3VAESidTest(unittest.TestCase):
    @staticmethod
    def items() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "item_id": list(range(8)),
                "normalized_embed": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.1, 0.9, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.9, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.1, 0.0, 0.0, 0.9],
                ],
            }
        )

    @staticmethod
    def config() -> R3VAEConfig:
        return R3VAEConfig(
            codebook_sizes=(4, 4),
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            kmeans_iterations=2,
            seed=11,
            device="cpu",
        )

    def test_pipeline_trains_and_emits_integer_sid_columns(self) -> None:
        sid = R3VAESid(self.config())
        sid.train(self.items())
        result = sid.update(self.items())

        self.assertEqual(result.columns, ["item_id", "sid_0", "sid_1"])
        self.assertEqual(result.height, 8)
        self.assertTrue(result.schema["sid_0"].is_integer())
        self.assertTrue(result.schema["sid_1"].is_integer())
        self.assertEqual(len(sid.training_history), 2)
        for epoch in sid.training_history:
            self.assertTrue(all(math.isfinite(value) for value in epoch.values()))

    def test_training_is_reproducible(self) -> None:
        first = R3VAESid(self.config())
        second = R3VAESid(self.config())
        first.train(self.items())
        second.train(self.items())

        self.assertEqual(first.training_history, second.training_history)
        self.assertEqual(
            first.update(self.items()).to_dict(as_series=False),
            second.update(self.items()).to_dict(as_series=False),
        )

    def test_zero_vectors_are_rejected(self) -> None:
        items = pl.DataFrame(
            {"item_id": [0], "normalized_embed": [[0.0, 0.0]]}
        )
        with self.assertRaisesRegex(ValueError, "zero vectors"):
            R3VAESid(self.config()).train(items)


if __name__ == "__main__":
    unittest.main()
