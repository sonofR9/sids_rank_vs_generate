from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.datasets import DatasetFiles
from sids_rank_vs_generate.sid_experiment import (
    SidExperimentConfig,
    run_sid_experiment,
)
from sids_rank_vs_generate.sids import (
    CollaborativeAlignmentConfig,
    CollaborativeAwareRQVAEConfig,
    CollaborativeAwareRQVAEPipeline,
    CollaborativeAwareRQKMeansConfig,
    CollaborativeAwareRQKMeansPipeline,
    SidTrainingData,
    mine_collaborative_pairs,
)


def sample_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": list(range(6)),
            "normalized_embed": [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.8, 0.2, 0.0],
            ],
        }
    )


def sample_train_interactions() -> pl.DataFrame:
    rows: list[tuple[int, int, int]] = []
    histories = {
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 4],
        3: [0, 1, 2, 5],
        4: [3, 4, 5, 2],
    }
    for uid, items in histories.items():
        rows.extend((uid, item_id, position) for position, item_id in enumerate(items))
    return pl.DataFrame(
        rows,
        schema=["uid", "item_id", "timestamp"],
        orient="row",
    )


class CollaborativePairMiningTest(unittest.TestCase):
    def test_pair_support_counts_distinct_train_users(self) -> None:
        interactions = pl.concat(
            [
                sample_train_interactions(),
                pl.DataFrame(
                    {"uid": [1], "item_id": [1], "timestamp": [1]}
                ),
            ]
        )
        config = CollaborativeAlignmentConfig(
            history_window=1,
            min_pair_support=1,
            pairs_per_item=10,
            max_pairs=None,
            epochs=1,
        )

        pairs = mine_collaborative_pairs(interactions, config)

        support = {
            (row["item_i"], row["item_j"]): row["support"]
            for row in pairs.iter_rows(named=True)
        }
        self.assertEqual(support[(0, 1)], 3)
        self.assertEqual(support[(1, 2)], 3)
        self.assertTrue(pairs.equals(mine_collaborative_pairs(interactions, config)))

    def test_training_contract_rejects_future_interactions(self) -> None:
        interactions = sample_train_interactions().with_columns(
            pl.when(pl.col("uid") == 4)
            .then(100)
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )

        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            SidTrainingData(
                item_features=sample_features(),
                train_interactions=interactions,
                train_end_timestamp=100,
            )


class CollaborativeAwareRQKMeansPipelineTest(unittest.TestCase):
    def test_aligns_train_pairs_before_rq_kmeans(self) -> None:
        alignment = CollaborativeAlignmentConfig(
            history_window=2,
            min_pair_support=1,
            pairs_per_item=4,
            max_pairs=20,
            hidden_dim=8,
            batch_size=4,
            inference_batch_size=3,
            epochs=2,
            learning_rate=1e-2,
            content_preservation_weight=0.5,
            device="cpu",
        )
        pipeline = CollaborativeAwareRQKMeansPipeline(
            CollaborativeAwareRQKMeansConfig(
                num_levels=2,
                codebook_size=2,
                kmeans_batch_size=8,
                predict_batch_size=3,
                alignment=alignment,
            )
        )
        data = SidTrainingData(
            item_features=sample_features(),
            train_interactions=sample_train_interactions(),
            train_end_timestamp=10,
        )

        pipeline.train_with_context(data)
        output = pipeline.update(sample_features())

        self.assertEqual(output.columns, ["item_id", "sid_0", "sid_1"])
        self.assertEqual(output.height, sample_features().height)
        self.assertTrue(output.schema["sid_0"].is_integer())
        summary = pipeline.summary()
        assert summary is not None
        self.assertEqual(summary["pair_mining"]["source"], "train_interactions")
        self.assertGreaterEqual(summary["pair_mining"]["pair_count"], 2)
        self.assertGreater(summary["alignment"]["mean_content_cosine"], 0.0)

    def test_plain_train_cannot_silently_skip_collaborative_data(self) -> None:
        pipeline = CollaborativeAwareRQKMeansPipeline()

        with self.assertRaisesRegex(RuntimeError, "requires train interactions"):
            pipeline.train(sample_features())

    def test_common_runner_supplies_train_only_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            original_ids = [10, 20, 30, 40, 50, 60]
            histories = {
                1: [10, 20, 30],
                2: [10, 20, 30],
                3: [40, 50, 60],
                4: [40, 50, 60],
            }
            rows = [
                (uid, item_id, timestamp)
                for uid, items in histories.items()
                for timestamp, item_id in enumerate(items)
            ]
            rows.extend((uid, 60, 4) for uid in histories)
            interactions_path = directory / "interactions.parquet"
            embeddings_path = directory / "embeddings.parquet"
            pl.DataFrame(
                rows,
                schema=["uid", "item_id", "timestamp"],
                orient="row",
            ).write_parquet(interactions_path)
            sample_features().with_columns(
                pl.Series("item_id", original_ids)
            ).write_parquet(embeddings_path)
            pipeline = CollaborativeAwareRQKMeansPipeline(
                CollaborativeAwareRQKMeansConfig(
                    num_levels=2,
                    codebook_size=2,
                    alignment=CollaborativeAlignmentConfig(
                        history_window=1,
                        min_pair_support=2,
                        pairs_per_item=4,
                        max_pairs=20,
                        hidden_dim=8,
                        batch_size=4,
                        epochs=1,
                        device="cpu",
                    ),
                )
            )

            result = run_sid_experiment(
                SidExperimentConfig(
                    dataset=DatasetFiles(
                        size="50m",
                        interactions_path=interactions_path,
                        embeddings_path=embeddings_path,
                    ),
                    sid_pipeline=pipeline,
                    reports_dir=directory / "reports",
                    core_min_interactions_per_item=1,
                    test_days=1,
                    timestamp_unit_seconds=86_400,
                )
            )

            details = result["sid"]["details"]
            self.assertEqual(details["pair_mining"]["source"], "train_interactions")
            self.assertEqual(details["pair_mining"]["interaction_count"], 12)
            self.assertEqual(details["pair_mining"]["train_end_timestamp"], "3.0")
            self.assertEqual(result["sid"]["item_count"], len(original_ids))


class CollaborativeAwareRQVAEPipelineTest(unittest.TestCase):
    def test_aligned_embeddings_feed_rqvae(self) -> None:
        pipeline = CollaborativeAwareRQVAEPipeline(
            CollaborativeAwareRQVAEConfig(
                hidden_dim=4,
                latent_dim=2,
                num_levels=2,
                codebook_size=2,
                batch_size=3,
                num_workers=0,
                pretrain_epochs=1,
                max_epochs=1,
                patience=1,
                val_part=0.2,
                psi_top_k=2,
                alignment=CollaborativeAlignmentConfig(
                    history_window=2,
                    min_pair_support=1,
                    pairs_per_item=4,
                    max_pairs=20,
                    hidden_dim=8,
                    batch_size=4,
                    inference_batch_size=3,
                    epochs=1,
                    learning_rate=1e-2,
                    device="cpu",
                ),
            )
        )
        data = SidTrainingData(
            item_features=sample_features(),
            train_interactions=sample_train_interactions(),
            train_end_timestamp=10,
        )

        pipeline.train_with_context(data)
        output = pipeline.update(sample_features())

        self.assertEqual(output.columns, ["item_id", "sid_0", "sid_1"])
        self.assertEqual(output.height, sample_features().height)
        summary = pipeline.summary()
        self.assertEqual(summary["pair_mining"]["source"], "train_interactions")
        self.assertEqual(summary["tokenizer"]["type"], "rqvae")


if __name__ == "__main__":
    unittest.main()
