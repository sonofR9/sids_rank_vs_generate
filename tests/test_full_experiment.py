from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import polars as pl

from sids_rank_vs_generate.data.ranking import DENSE_COLUMNS
from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.evaluation.ranking_stage import RankingConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
    run_experiment,
)
from sids_rank_vs_generate.sids import (
    RQKMeansConfig,
    RQKMeansPipeline,
    UserAwareRQKMeansConfig,
    UserAwareRQKMeansPipeline,
    UserSidMapperConfig,
)
from sids_rank_vs_generate.tracking import ExperimentTracker


class RecordingTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.runs = {}
        self.stack = []

    @contextmanager
    def start_run(self, run_name, *, nested=False, tags=None):
        self.runs[run_name] = {
            "nested": nested,
            "tags": dict(tags or {}),
            "params": {},
            "metrics": [],
        }
        self.stack.append(run_name)
        try:
            yield self
        finally:
            self.stack.pop()

    def log_params(self, params) -> None:
        self.runs[self.stack[-1]]["params"].update(params)

    def log_metrics(self, metrics, *, step=None) -> None:
        self.runs[self.stack[-1]]["metrics"].append((step, dict(metrics)))

    def info(self):
        return {
            "enabled": True,
            "run_id": self.stack[0] if self.stack else None,
        }


class FullRQKMeansExperimentTest(unittest.TestCase):
    def test_sid_and_experiment_configs_do_not_accept_seed_overrides(self) -> None:
        with self.assertRaises(TypeError):
            RQKMeansConfig(random_state=7)
        with self.assertRaises(TypeError):
            ExperimentConfig(
                sid_pipeline=RQKMeansPipeline(RQKMeansConfig()),
                seed=7,
            )

    def test_user_aware_experiment_requires_fit_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, "catalog_from_fit"):
            ExperimentConfig(
                sid_pipeline=RQKMeansPipeline(RQKMeansConfig()),
                user_sid_pipeline=UserAwareRQKMeansPipeline(),
            )

    def test_rqkmeans_runs_ranking_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            interactions_path = root / "interactions.parquet"
            embeddings_path = root / "embeddings.parquet"
            ranking_path = root / "ranking.parquet"
            artist_path = root / "artists.parquet"
            album_path = root / "albums.parquet"

            rows = []
            for uid in range(4):
                for day in range(12):
                    rows.append(
                        {
                            "uid": uid,
                            "item_id": (day + uid) % 6,
                            "timestamp": day * 86_400 + uid,
                        }
                    )
            pl.DataFrame(rows).write_parquet(interactions_path)
            pl.DataFrame(
                {
                    "item_id": list(range(6)),
                    "normalized_embed": [
                        [float(item == dimension) for dimension in range(6)]
                        for item in range(6)
                    ],
                }
            ).write_parquet(embeddings_path)
            ranking_rows = []
            for uid in range(4):
                for day in range(1, 12):
                    row = {
                        "uid": uid,
                        "item_id": (day + uid + 1) % 6,
                        "timestamp": day * 86_400 + uid + 100,
                        "played_ratio_pct": 100 if (day + uid) % 2 else 25,
                        "track_length_seconds": 180 + day,
                        "is_like": bool((day + uid) % 2),
                    }
                    row.update(
                        {
                            column: float(day + uid + feature_index + 1)
                            for feature_index, column in enumerate(DENSE_COLUMNS)
                        }
                    )
                    ranking_rows.append(row)
            pl.DataFrame(ranking_rows).write_parquet(ranking_path)
            pl.DataFrame(
                {
                    "artist_id": [100 + item for item in range(6)],
                    "item_id": list(range(6)),
                }
            ).write_parquet(artist_path)
            pl.DataFrame(
                {
                    "album_id": [200 + item for item in range(6)],
                    "item_id": list(range(6)),
                }
            ).write_parquet(album_path)

            config = ExperimentConfig(
                dataset=YambdaDatasetConfig(
                    interactions_path=interactions_path,
                    embeddings_path=embeddings_path,
                    ranking_interactions_path=ranking_path,
                    artist_mapping_path=artist_path,
                    album_mapping_path=album_path,
                    core_min_interactions_per_item=1,
                    test_days=3,
                    timestamp_unit_seconds=1,
                ),
                sid_pipeline=RQKMeansPipeline(
                    RQKMeansConfig(num_levels=2, codebook_size=2)
                ),
                training=TrainingConfig(
                    ranking=RankingConfig(
                        max_history_len=4,
                        min_history_len=1,
                        batch_size=16,
                        epochs=1,
                        embedding_dim=4,
                        hidden_dim=8,
                        dense_bins=2,
                        dcn_cross_layers=1,
                        dcn_deep_units=(16, 8),
                        dcn_num_experts=2,
                        dcn_low_rank=4,
                        transact_d_model=8,
                        transact_heads=2,
                        transact_layers=1,
                        transact_ff_dim=16,
                        dropout=0.0,
                        show_progress=False,
                        device="cpu",
                    ),
                    generation=GenerationConfig(
                        max_history_len=4,
                        min_history_len=1,
                        d_model=8,
                        d_ff=16,
                        num_heads=2,
                        num_layers=1,
                        dropout=0.0,
                        batch_size=16,
                        batch_size_val=16,
                        num_workers=0,
                        max_epochs=1,
                        num_beams=4,
                        recall_k=2,
                        device="cpu",
                    ),
                ),
                catalog_from_fit=True,
                reports_dir=root / "reports",
            )

            tracker = RecordingTracker()
            with patch(
                "sids_rank_vs_generate.experiment.create_tracker",
                return_value=tracker,
            ):
                result = run_experiment(config)

            self.assertEqual(set(result["evaluations"]), {"ranking", "generation"})
            self.assertEqual(
                set(result["evaluations"]["ranking"]["models"]),
                {"dcnv2", "transact"},
            )
            self.assertNotIn(
                "history",
                result["evaluations"]["ranking"]["models"]["dcnv2"],
            )
            self.assertNotIn(
                "history",
                result["evaluations"]["ranking"]["models"]["transact"],
            )
            ranking_features = result["evaluations"]["ranking"]["features"]
            self.assertEqual(ranking_features["dense"], list(DENSE_COLUMNS))
            self.assertEqual(
                ranking_features["multivalent"], ["artist_ids", "album_ids"]
            )
            self.assertIn("statistics", result["sid"])
            self.assertNotIn("history", result["evaluations"]["generation"])
            self.assertTrue(Path(result["artifacts"]["semantic_ids"]).is_file())
            self.assertTrue(
                Path(result["evaluations"]["generation"]["checkpoint"]).is_file()
            )
            self.assertEqual(
                set(tracker.runs),
                {
                    "RQKMeansPipeline",
                    "sid",
                    "ranking.dcnv2",
                    "ranking.transact",
                    "generation",
                },
            )
            for run_name in ("ranking.dcnv2", "ranking.transact", "generation"):
                metric_names = {
                    name
                    for _step, metrics in tracker.runs[run_name]["metrics"]
                    for name in metrics
                }
                self.assertIn("train.loss", metric_names)

            user_tracker = RecordingTracker()
            with patch(
                "sids_rank_vs_generate.experiment.create_tracker",
                return_value=user_tracker,
            ):
                user_aware_result = run_experiment(
                    replace(
                        config,
                        user_sid_pipeline=UserAwareRQKMeansPipeline(
                            UserAwareRQKMeansConfig(
                                num_levels=2,
                                codebook_size=2,
                                kmeans_batch_size=16,
                                predict_batch_size=16,
                            )
                        ),
                        training=replace(
                            config.training,
                            generation=replace(
                                config.training.generation,
                                mapper=UserSidMapperConfig(
                                    kind="transformer",
                                    d_model=8,
                                    d_ff=16,
                                    num_heads=2,
                                    num_encoder_layers=1,
                                    num_decoder_layers=1,
                                    dropout=0.0,
                                    batch_size=16,
                                    inference_batch_size=16,
                                    max_epochs=1,
                                    early_stopping_patience=1,
                                ),
                            ),
                        ),
                    )
                )

            self.assertEqual(
                user_aware_result["evaluations"]["generation"]["target_sid_type"],
                "personalized",
            )
            self.assertIsNotNone(user_aware_result["user_sid"])
            self.assertTrue(
                all(
                    column.startswith("usid_")
                    for column in user_aware_result["evaluations"]["ranking"][
                        "features"
                    ]["sparse"][1:]
                )
            )
            self.assertIn(
                "test",
                user_aware_result["evaluations"]["generation"]["mapper"],
            )
            self.assertEqual(
                user_aware_result["evaluations"]["generation"]["mapper"]["type"],
                "transformer",
            )
            self.assertTrue(
                Path(
                    user_aware_result["evaluations"]["generation"]["mapper"][
                        "checkpoint"
                    ]
                ).is_file()
            )
            self.assertEqual(
                set(user_tracker.runs),
                {
                    "RQKMeansPipeline",
                    "sid",
                    "user_sid",
                    "ranking.dcnv2",
                    "ranking.transact",
                    "generation",
                    "generation.mapper",
                },
            )
            mapper_metric_names = {
                name
                for _step, metrics in user_tracker.runs["generation.mapper"]["metrics"]
                for name in metrics
            }
            self.assertIn("train.loss", mapper_metric_names)


if __name__ == "__main__":
    unittest.main()
