from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from sids_rank_vs_generate.experiment import load_experiment_config
from sids_rank_vs_generate.sids.balance_rq_kmeans_pipeline import (
    BalanceRQKMeansPipeline,
    _model_parameters as balance_model_parameters,
)
from sids_rank_vs_generate.sids.config import (
    BalanceRQKMeansConfig,
    PlumRQVAEConfig,
)
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import (
    PlumRQVAEPipeline,
    _model_parameters as plum_model_parameters,
)


CONFIG_PATH = Path("experiments/balance_plum_poster.py")


class BalancePlumPosterConfigTest(unittest.TestCase):
    def load(self, **environment: str):
        clean_environment = {
            "SID_MODEL": "balance",
            "COLLISION_RESOLUTION": "none",
            "NUM_LEVELS": "3",
            "CODEBOOK_SIZE": "512",
        }
        clean_environment.update(environment)
        with patch.dict(os.environ, clean_environment, clear=True):
            return load_experiment_config(str(CONFIG_PATH))

    def test_balance_configuration_uses_uniform_codebooks(self) -> None:
        experiment = self.load(SID_MODEL="balance", CODEBOOK_SIZE="128")
        self.assertIsInstance(experiment.sid_pipeline, BalanceRQKMeansPipeline)
        self.assertEqual(experiment.sid_pipeline.config.num_levels, 3)
        self.assertEqual(experiment.sid_pipeline.config.codebook_size, 128)
        self.assertFalse(experiment.training.generation.resolve_collisions)

    def test_plum_configuration_uses_requested_codebooks(self) -> None:
        experiment = self.load(
            SID_MODEL="plum",
            CODEBOOK_SIZES="1024,512,256",
        )
        self.assertIsInstance(experiment.sid_pipeline, PlumRQVAEPipeline)
        self.assertEqual(
            experiment.sid_pipeline.config.codebook_sizes,
            (1024, 512, 256),
        )
        self.assertEqual(experiment.sid_pipeline.config.num_levels, 3)

    def test_only_none_collision_mode_is_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "none only"):
            self.load(COLLISION_RESOLUTION="token")

    def test_unknown_sid_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "SID_MODEL"):
            self.load(SID_MODEL="unknown")

    def test_pipeline_parameters_translate_auto_device_and_codebook_names(self) -> None:
        balance = balance_model_parameters(
            BalanceRQKMeansConfig(num_levels=3, codebook_size=128, device="auto")
        )
        self.assertIsNone(balance["device"])
        self.assertEqual(balance["num_levels"], 3)
        self.assertEqual(balance["codebook_size"], 128)

        plum = plum_model_parameters(
            PlumRQVAEConfig(
                num_levels=3,
                codebook_sizes=(256, 128, 64),
                device="auto",
            )
        )
        self.assertIsNone(plum["device"])
        self.assertEqual(plum["codebooks_sizes"], (256, 128, 64))
        self.assertNotIn("codebook_sizes", plum)


class BalancePlumPipelineSmokeTest(unittest.TestCase):
    @staticmethod
    def item_features() -> pl.DataFrame:
        generator = np.random.default_rng(42)
        embeddings = generator.normal(size=(12, 6)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return pl.DataFrame(
            {
                "item_id": np.arange(len(embeddings)),
                "normalized_embed": embeddings.tolist(),
            }
        )

    def test_balance_pipeline_trains_and_transforms(self) -> None:
        features = self.item_features()
        pipeline = BalanceRQKMeansPipeline(
            BalanceRQKMeansConfig(
                num_levels=2,
                codebook_size=2,
                max_iter=2,
                device="cpu",
            )
        )
        pipeline.train(features)
        result = pipeline.update(features)
        self.assertEqual(result.shape, (12, 3))

    def test_plum_pipeline_trains_and_transforms(self) -> None:
        features = self.item_features()
        pipeline = PlumRQVAEPipeline(
            PlumRQVAEConfig(
                hidden_dim=4,
                latent_dim=4,
                num_levels=2,
                codebook_size=4,
                codebook_sizes=(4, 2),
                batch_size=4,
                num_epochs=1,
                num_workers=0,
                device="cpu",
                pretrain_epochs=1,
            )
        )
        pipeline.train(features)
        result = pipeline.update(features)
        self.assertEqual(result.shape, (12, 3))


if __name__ == "__main__":
    unittest.main()
