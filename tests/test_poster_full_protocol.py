from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from sids_rank_vs_generate.data import (
    POSTER_FULL_33148_DATA_CONTRACT,
    POSTER_FULL_33148_PROTOCOL_ID,
    PreprocessingConfig,
    prepare_interactions,
    validate_poster_full_33148_summary,
)
from sids_rank_vs_generate.data.generation import prepare_generation_examples
from sids_rank_vs_generate.datasets import DatasetFiles
from sids_rank_vs_generate.sid_experiment import (
    SidExperimentConfig,
    run_sid_experiment,
)
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


DAY_BIN = 24 * 60 * 60 // 5


def poster_sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1] * 16,
            "item_id": [10] * 11 + [20] * 5,
            "timestamp": [day * DAY_BIN for day in range(11)]
            + [day * DAY_BIN + 1 for day in range(6, 11)],
        }
    )


def poster_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        core_min_interactions_per_item=5,
        history_days=None,
        test_days=7,
        timestamp_unit_seconds=5,
        catalog_from_fit=False,
        protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
    )


class PosterFullProtocolTest(unittest.TestCase):
    def test_legacy_heldout_is_used_for_validation_and_reporting(self) -> None:
        prepared = prepare_interactions(poster_sample(), poster_config())

        self.assertEqual(prepared.catalog_size, 2)
        self.assertTrue(prepared.fit.equals(prepared.train))
        self.assertTrue(prepared.validation.equals(prepared.test))
        self.assertEqual(
            prepared.validation_start_timestamp,
            prepared.test_start_timestamp,
        )
        self.assertEqual(
            prepared.summary()["protocol_id"],
            POSTER_FULL_33148_PROTOCOL_ID,
        )

    def test_generation_duplicates_heldout_examples(self) -> None:
        prepared = prepare_interactions(poster_sample(), poster_config())
        semantic_ids = pl.DataFrame(
            {
                "item_id": [0, 1],
                "sid_0": [0, 1],
            }
        )
        config = SimpleNamespace(
            num_levels=1,
            min_history_len=1,
            max_history_len=16,
        )

        train, validation, test, _item2idx = prepare_generation_examples(
            prepared,
            semantic_ids,
            config,
        )

        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertEqual(validation, test)

    def test_heldout_boundary_is_computed_after_item_core(self) -> None:
        rare_latest = pl.DataFrame(
            {
                "uid": [2],
                "item_id": [999],
                "timestamp": [100 * DAY_BIN],
            }
        )

        prepared = prepare_interactions(
            pl.concat([poster_sample(), rare_latest]),
            poster_config(),
        )

        expected_boundary = poster_sample()["timestamp"].max() - 7 * DAY_BIN
        self.assertEqual(prepared.test_start_timestamp, expected_boundary)
        self.assertNotIn(999, prepared.item_mapping["original_item_id"].to_list())
        self.assertFalse(prepared.test.is_empty())

    def test_poster_protocol_rejects_different_data_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "full history"):
            PreprocessingConfig(
                history_days=365,
                timestamp_unit_seconds=5,
                protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
            )
        with self.assertRaisesRegex(ValueError, "full-data catalog"):
            PreprocessingConfig(
                history_days=None,
                timestamp_unit_seconds=5,
                catalog_from_fit=True,
                protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
            )

    def test_data_contract_rejects_same_catalog_with_wrong_split(self) -> None:
        summary = dict(POSTER_FULL_33148_DATA_CONTRACT)
        summary["test_interactions"] = 4_546

        with self.assertRaisesRegex(
            ValueError,
            "data mismatch for test_interactions",
        ):
            validate_poster_full_33148_summary(summary)

    def test_runner_rejects_wrong_catalog_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            interactions_path = root / "interactions.parquet"
            embeddings_path = root / "embeddings.parquet"
            poster_sample().write_parquet(interactions_path)
            pl.DataFrame(
                {
                    "item_id": [10, 20],
                    "normalized_embed": [[1.0, 0.0], [0.0, 1.0]],
                }
            ).write_parquet(embeddings_path)

            with self.assertRaisesRegex(
                ValueError,
                "expected 33148, received 2",
            ):
                run_sid_experiment(
                    SidExperimentConfig(
                        dataset=DatasetFiles(
                            size="50m",
                            interactions_path=interactions_path,
                            embeddings_path=embeddings_path,
                        ),
                        sid_pipeline=RQKMeansPipeline(
                            RQKMeansConfig(num_levels=1, codebook_size=2)
                        ),
                        core_min_interactions_per_item=5,
                        history_days=None,
                        test_days=7,
                        timestamp_unit_seconds=5,
                        catalog_from_fit=False,
                        protocol_id=POSTER_FULL_33148_PROTOCOL_ID,
                    )
                )


if __name__ == "__main__":
    unittest.main()
