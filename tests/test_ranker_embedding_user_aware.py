from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.ranker_embedding_user_aware import (
    RankerEmbeddingUserAwareRQKMeansConfig,
    RankerEmbeddingUserAwareRQKMeansPipeline,
)
from sids_rank_vs_generate.sids.user_aware import UserAwareSidTrainingData
from sids_rank_vs_generate.training.ranker_pair_encoder import (
    RankerPairEncoderTrainingConfig,
)


def _training_data(*, include_behavior: bool = True):
    base = UserAwareSidTrainingData(
        fit_interactions=pl.DataFrame(
            {
                "uid": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                "item_id": [10, 11, 12, 10, 11, 12, 10, 11, 12, 10],
                "timestamp": list(range(1, 11)),
            }
        ),
        item_features=pl.DataFrame(
            {
                "item_id": [10, 11, 12],
                "normalized_embed": [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.7071068, 0.7071068],
                ],
            }
        ),
        global_sids=pl.DataFrame(
            {
                "item_id": [10, 11, 12],
                "sid_0": [0, 1, 0],
                "sid_1": [1, 0, 0],
            }
        ),
        fit_end_timestamp=11,
    )
    values = {
        "fit_interactions": base.fit_interactions,
        "item_features": base.item_features,
        "global_sids": base.global_sids,
        "fit_end_timestamp": base.fit_end_timestamp,
    }
    if include_behavior:
        values["fit_behavior_interactions"] = pl.DataFrame(
            {
                "uid": [1, 2, 1, 2, 1, 2, 1, 2],
                "item_id": [10, 11, 12, 10, 11, 12, 10, 11],
                "timestamp": [5, 5, 6, 7, 8, 9, 10, 10],
                "target": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            }
        )
    return SimpleNamespace(**values)


def _config() -> RankerEmbeddingUserAwareRQKMeansConfig:
    return RankerEmbeddingUserAwareRQKMeansConfig(
        num_levels=2,
        codebook_size=2,
        kmeans_batch_size=8,
        predict_batch_size=8,
        ranker_hidden_dims=(8,),
        ranker_representation_dim=4,
        ranker_dropout=0.0,
        fusion_weight=0.35,
        ranker_label_fraction=0.5,
        negative_ratio=2,
        ranker_training=RankerPairEncoderTrainingConfig(
            batch_size=4,
            epochs=1,
            learning_rate=0.01,
            device="cpu",
            precision="fp32",
            seed=13,
            show_progress=False,
        ),
    )


class RankerEmbeddingUserAwarePipelineTest(unittest.TestCase):
    def test_negative_ratio_requires_an_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            RankerEmbeddingUserAwareRQKMeansConfig(
                negative_ratio=1.5  # type: ignore[arg-type]
            )

    def test_pipeline_declares_behavior_requirement(self) -> None:
        pipeline = RankerEmbeddingUserAwareRQKMeansPipeline(_config())

        self.assertTrue(pipeline.requires_behavior_interactions)

    def test_missing_behavior_is_rejected(self) -> None:
        pipeline = RankerEmbeddingUserAwareRQKMeansPipeline(_config())

        with self.assertRaisesRegex(ValueError, "fit_behavior_interactions"):
            pipeline.train(_training_data(include_behavior=False))

    def test_cutoff_separates_profiles_and_fit_labels(self) -> None:
        pipeline = RankerEmbeddingUserAwareRQKMeansPipeline(_config())

        pipeline.train(_training_data())
        summary = pipeline.training_summary

        self.assertLess(
            summary["ranker_profile_max_timestamp"],
            summary["ranker_cutoff"],
        )
        self.assertGreaterEqual(
            summary["ranker_label_min_timestamp"],
            summary["ranker_cutoff"],
        )
        self.assertLess(summary["ranker_label_max_timestamp"], 11)
        self.assertEqual(summary["ranker_rows"], 6)
        self.assertEqual(summary["ranker_positives"], 3)
        self.assertEqual(summary["ranker_negatives"], 3)
        self.assertEqual(summary["ranker_unknown_profile_rows"], 0)
        self.assertEqual(summary["ranker_unknown_profile_rate"], 0.0)

    def test_fit_and_sid_assignment_are_deterministic(self) -> None:
        data = _training_data()
        pairs = pl.DataFrame(
            {"uid": [1, 1, 2, 999], "item_id": [10, 12, 11, 10]}
        )
        first = RankerEmbeddingUserAwareRQKMeansPipeline(_config())
        second = RankerEmbeddingUserAwareRQKMeansPipeline(_config())

        first.train(data)
        second.train(data)

        self.assertTrue(first.update(pairs).equals(second.update(pairs)))

    def test_fusion_is_normalized_and_keeps_both_representations(self) -> None:
        pipeline = RankerEmbeddingUserAwareRQKMeansPipeline(_config())
        pipeline.train(_training_data())
        pairs = pl.DataFrame({"uid": [1, 2], "item_id": [10, 12]})

        features = pipeline._pair_features(pairs)

        self.assertEqual(features.shape, (2, 12))
        self.assertTrue(
            np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-6)
        )
        self.assertTrue(np.any(np.abs(features[:, :8]) > 0))
        self.assertTrue(np.any(np.abs(features[:, 8:]) > 0))

    def test_validation_or_test_behavior_is_rejected(self) -> None:
        data = _training_data()
        data.fit_behavior_interactions = pl.concat(
            [
                data.fit_behavior_interactions,
                pl.DataFrame(
                    {
                        "uid": [1],
                        "item_id": [10],
                        "timestamp": [11],
                        "target": [1.0],
                    }
                ),
            ]
        )
        pipeline = RankerEmbeddingUserAwareRQKMeansPipeline(_config())

        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            pipeline.train(data)


if __name__ == "__main__":
    unittest.main()
