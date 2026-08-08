from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl

from sids_rank_vs_generate.sids import (
    FrequencyUserSidMapper,
    TransformerUserSidMapper,
    UserAwareRQKMeansConfig,
    UserAwareRQKMeansPipeline,
    UserAwareSidTrainingData,
    UserSidMapperConfig,
    user_sid_columns,
    validate_user_sid_output,
)
from sids_rank_vs_generate.evaluation.user_sid_statistics import (
    calculate_user_sid_statistics,
)


def training_data(**overrides) -> UserAwareSidTrainingData:
    values = {
        "fit_interactions": pl.DataFrame(
            {
                "uid": [1, 1, 2],
                "item_id": [10, 11, 10],
                "timestamp": [1, 2, 3],
            }
        ),
        "item_features": pl.DataFrame(
            {
                "item_id": [10, 11],
                "normalized_embed": [[1.0, 0.0], [0.0, 1.0]],
            }
        ),
        "global_sids": pl.DataFrame(
            {
                "item_id": [10, 11],
                "sid_0": [0, 1],
                "sid_1": [1, 0],
            }
        ),
        "fit_end_timestamp": 4,
    }
    values.update(overrides)
    return UserAwareSidTrainingData(**values)


class UserAwareSidContractTest(unittest.TestCase):
    def test_training_data_accepts_strictly_train_only_rows(self) -> None:
        data = training_data()
        self.assertEqual(data.fit_interactions.height, 3)

    def test_training_data_rejects_boundary_or_future_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            training_data(fit_end_timestamp=3)

    def test_training_data_requires_matching_catalogs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same catalog"):
            training_data(global_sids=pl.DataFrame({"item_id": [10], "sid_0": [0]}))

    def test_user_sid_columns_are_sorted_numerically(self) -> None:
        frame = pl.DataFrame(
            {
                "uid": [1],
                "item_id": [10],
                "usid_10": [2],
                "usid_2": [1],
                "usid_0": [0],
            }
        )
        self.assertEqual(
            user_sid_columns(frame),
            ["usid_0", "usid_2", "usid_10"],
        )

    def test_validate_user_sid_output_accepts_matching_integer_codes(self) -> None:
        pairs = pl.DataFrame({"uid": [1, 2], "item_id": [10, 11]})
        output = pairs.with_columns(
            pl.Series("usid_0", [0, 1], dtype=pl.Int32),
            pl.Series("usid_1", [1, 0], dtype=pl.Int32),
        )
        self.assertEqual(
            validate_user_sid_output(pairs, output),
            ["usid_0", "usid_1"],
        )

    def test_validate_user_sid_output_rejects_pair_mismatch(self) -> None:
        pairs = pl.DataFrame({"uid": [1, 2], "item_id": [10, 11]})
        output = pl.DataFrame(
            {
                "uid": [1, 3],
                "item_id": [10, 11],
                "usid_0": [0, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_user_sid_output(pairs, output)

    def test_user_aware_rqkmeans_is_deterministic_and_supports_unknown_users(
        self,
    ) -> None:
        data = training_data()
        pairs = pl.DataFrame(
            {
                "uid": [1, 2, 999],
                "item_id": [10, 11, 10],
            }
        )
        config = UserAwareRQKMeansConfig(
            num_levels=2,
            codebook_size=2,
            kmeans_batch_size=4,
            predict_batch_size=4,
        )
        first = UserAwareRQKMeansPipeline(config)
        second = UserAwareRQKMeansPipeline(config)
        first.train(data)
        second.train(data)

        self.assertTrue(first.update(pairs).equals(second.update(pairs)))
        self.assertEqual(first.user_features(pairs.select("uid")).height, 3)

    def test_user_aware_rqkmeans_accepts_per_level_codebook_sizes(self) -> None:
        pipeline = UserAwareRQKMeansPipeline(
            UserAwareRQKMeansConfig(
                codebook_sizes=(2, 2),
                kmeans_batch_size=4,
                predict_batch_size=4,
            )
        )
        pipeline.train(training_data())

        result = pipeline.update(pl.DataFrame({"uid": [1], "item_id": [10]}))

        self.assertEqual(user_sid_columns(result), ["usid_0", "usid_1"])

    def test_frequency_mapper_uses_user_then_global_fallback(self) -> None:
        mapper = FrequencyUserSidMapper()
        mapper.fit(
            pl.DataFrame(
                {
                    "uid": [1, 1, 2],
                    "usid_0": [0, 0, 0],
                    "sid_0": [1, 1, 2],
                }
            )
        )

        self.assertEqual(mapper.predict_topk(1, [0], 1), [(1,)])
        self.assertEqual(mapper.predict_topk(999, [0], 1), [(1,)])

    def test_frequency_mapper_fills_topk_from_fallbacks(self) -> None:
        mapper = FrequencyUserSidMapper()
        mapper.fit(
            pl.DataFrame(
                {
                    "uid": [1, 2, 3],
                    "usid_0": [0, 0, 1],
                    "sid_0": [1, 2, 3],
                }
            )
        )

        self.assertEqual(mapper.predict_topk(1, [0], 3), [(1,), (2,), (3,)])

    def test_transformer_mapper_returns_only_catalog_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_paths = {(0, 0), (0, 1), (1, 1)}
            mapper = TransformerUserSidMapper(
                UserSidMapperConfig(
                    kind="transformer",
                    d_model=8,
                    d_ff=16,
                    num_heads=2,
                    num_encoder_layers=1,
                    num_decoder_layers=1,
                    dropout=0.0,
                    batch_size=4,
                    inference_batch_size=4,
                    max_epochs=1,
                    early_stopping_patience=1,
                ),
                user_sid_sizes=(2, 2),
                global_sid_sizes=(2, 2),
                valid_global_sids=valid_paths,
                user_features=pl.DataFrame(
                    {
                        "uid": [1, 2],
                        "user_feature": [[1.0, 0.0], [0.0, 1.0]],
                    }
                ),
                device="cpu",
                checkpoint_path=Path(tmpdir) / "mapper.pt",
            )
            training = pl.DataFrame(
                {
                    "uid": [1, 1, 2, 2],
                    "usid_0": [0, 1, 0, 1],
                    "usid_1": [0, 1, 1, 0],
                    "sid_0": [0, 1, 0, 1],
                    "sid_1": [0, 1, 1, 1],
                }
            )
            validation = [
                {
                    "uid": 1,
                    "sid": [0, 0],
                    "global_sid": [0, 0],
                }
            ]

            mapper.fit(training, validation)
            predictions = mapper.predict_topk_batch(
                [(1, [0, 0]), (999, [1, 1])],
                k=3,
            )

            with patch.object(
                mapper,
                "_predict_chunk",
                wraps=mapper._predict_chunk,
            ) as predict_chunk:
                duplicate_predictions = mapper.predict_topk_batch(
                    [(1, [0, 0]), (1, [0, 0]), (999, [1, 1])],
                    k=3,
                )

            self.assertTrue(Path(tmpdir, "mapper.pt").is_file())
            self.assertTrue(predictions)
            self.assertTrue(
                all(path in valid_paths for ranked in predictions for path in ranked)
            )
            self.assertEqual(duplicate_predictions[0], duplicate_predictions[1])
            self.assertEqual(predict_chunk.call_count, 1)
            self.assertEqual(len(predict_chunk.call_args.args[0]), 2)

    def test_transformer_mapper_loads_checkpoint_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = UserSidMapperConfig(
                kind="transformer",
                d_model=8,
                d_ff=16,
                num_heads=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dropout=0.0,
                batch_size=4,
                inference_batch_size=4,
                max_epochs=1,
                early_stopping_patience=1,
            )
            features = pl.DataFrame(
                {
                    "uid": [1, 2],
                    "user_feature": [[1.0, 0.0], [0.0, 1.0]],
                }
            )
            training = pl.DataFrame(
                {
                    "uid": [1, 1, 2, 2],
                    "usid_0": [0, 1, 0, 1],
                    "usid_1": [0, 1, 1, 0],
                    "sid_0": [0, 1, 0, 1],
                    "sid_1": [0, 1, 1, 1],
                }
            )
            validation = [
                {"uid": 1, "sid": [0, 0], "global_sid": [0, 0]}
            ]
            constructor = {
                "config": config,
                "user_sid_sizes": (2, 2),
                "global_sid_sizes": (2, 2),
                "valid_global_sids": {(0, 0), (0, 1), (1, 1)},
                "user_features": features,
                "device": "cpu",
            }
            trained = TransformerUserSidMapper(
                **constructor,
                checkpoint_path=root / "source.pt",
            )
            trained.fit(training, validation)
            expected = trained.predict_topk_batch(
                [(1, [0, 0]), (999, [1, 1])],
                k=3,
            )

            loaded = TransformerUserSidMapper(
                **constructor,
                checkpoint_path=root / "copied.pt",
            )
            loaded.load(training, root / "source.pt")

            self.assertEqual(
                loaded.predict_topk_batch(
                    [(1, [0, 0]), (999, [1, 1])],
                    k=3,
                ),
                expected,
            )
            self.assertTrue((root / "copied.pt").is_file())
            self.assertEqual(
                loaded.training_summary["loaded_checkpoint"],
                str(root / "source.pt"),
            )
            self.assertTrue(loaded.is_seen(1, [0, 0]))

    def test_user_sid_statistics_measure_cross_user_variation(self) -> None:
        frame = pl.DataFrame(
            {
                "uid": [1, 2, 1],
                "item_id": [10, 10, 11],
                "usid_0": [0, 1, 0],
                "usid_1": [1, 1, 0],
            }
        )

        statistics = calculate_user_sid_statistics(frame)

        self.assertEqual(statistics["pair_count"], 3)
        self.assertEqual(
            statistics["same_item_cross_user_variation"]["max_unique_paths"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
