from __future__ import annotations

import unittest

import polars as pl
import torch

from sids_rank_vs_generate.sids import RQVAEConfig, RQVAEPipeline
from sids_rank_vs_generate.sids.pure_semantic import allocate_semantic_ids


class PureSemanticAllocationTest(unittest.TestCase):
    @staticmethod
    def colliding_inputs() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        latents = torch.tensor(
            [[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]],
            dtype=torch.float32,
        )
        codebooks = (
            torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 2.0]]),
        )
        return latents, codebooks

    def test_rrs_resolves_collisions_without_extra_token(self) -> None:
        latents, codebooks = self.colliding_inputs()
        allocation = allocate_semantic_ids(
            latents,
            codebooks,
            method="rrs",
            top_k=2,
        )
        self.assertEqual(allocation.codes.shape, (3, 2))
        self.assertEqual(len(set(map(tuple, allocation.codes.tolist()))), 3)
        self.assertEqual(
            allocation.metrics["nearest_independent_coding_rate"],
            0.0,
        )
        self.assertEqual(
            allocation.metrics["final_independent_coding_rate"],
            1.0,
        )

    def test_ecm_resolves_collisions_without_extra_token(self) -> None:
        latents, codebooks = self.colliding_inputs()
        allocation = allocate_semantic_ids(
            latents,
            codebooks,
            method="ecm",
            top_k=2,
        )
        self.assertEqual(len(set(map(tuple, allocation.codes.tolist()))), 3)
        self.assertEqual(
            allocation.metrics["final_max_collision_bucket_size"],
            1,
        )

    def test_impossible_capacity_is_reported(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds code-space capacity"):
            allocate_semantic_ids(
                torch.zeros(5, 2),
                (torch.zeros(2, 2), torch.zeros(2, 2)),
                method="rrs",
                top_k=2,
            )


class RQVAEPipelinePsiTest(unittest.TestCase):
    @staticmethod
    def items() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "item_id": list(range(6)),
                "normalized_embed": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )

    def test_one_trained_model_supports_all_assignment_methods(self) -> None:
        pipeline = RQVAEPipeline(
            RQVAEConfig(
                hidden_dim=4,
                latent_dim=2,
                num_levels=2,
                codebook_size=3,
                batch_size=3,
                num_workers=0,
                pretrain_epochs=1,
                max_epochs=1,
                val_part=0.2,
                psi_top_k=3,
            )
        )
        pipeline.train(self.items())

        nearest = pipeline.assign(self.items(), method="nearest")
        rrs = pipeline.assign(self.items(), method="rrs")
        ecm = pipeline.assign(self.items(), method="ecm")

        self.assertEqual(nearest.columns, ["item_id", "sid_0", "sid_1"])
        for result in (rrs, ecm):
            codes = zip(result["sid_0"].to_list(), result["sid_1"].to_list())
            self.assertEqual(len(set(codes)), result.height)
        self.assertEqual(
            pipeline.summary()["assignment"]["final_independent_coding_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
