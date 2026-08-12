from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from sids_rank_vs_generate.cli import main
from sids_rank_vs_generate.datasets import DatasetFiles, resolve_dataset
from sids_rank_vs_generate.evaluation import (
    DEFAULT_STAGE_FACTORIES,
    EvaluationContext,
    IEvaluationStage,
)
from sids_rank_vs_generate.sid_experiment import (
    SidExperimentConfig,
    run_sid_experiment,
)
from sids_rank_vs_generate.sids import ISidPipeline
from tests.test_preprocessing import sample_interactions


class FakeSid(ISidPipeline):
    def __init__(self) -> None:
        self.trained = False

    def train(self, df: pl.DataFrame) -> None:
        if df["item_id"].to_list() != [0, 1, 2]:
            raise AssertionError("SID did not receive the shared dense item mapping")
        if "embed" not in df.columns:
            raise AssertionError("item feature columns were not preserved")
        self.trained = True

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.trained:
            raise AssertionError("train must run before update")
        return df.with_columns(
            (pl.col("item_id") % 2).cast(pl.Int32).alias("sid_0"),
            (pl.col("item_id") // 2).cast(pl.Int32).alias("sid_1"),
        )


class FakeRankingEvaluator(IEvaluationStage):
    @property
    def name(self) -> str:
        return "ranking"

    def evaluate(self, context: EvaluationContext) -> dict[str, float]:
        if "embed" not in context.items_with_sids.columns:
            raise AssertionError("original item features were not preserved")
        (context.artifact_dir / "curve.txt").write_text(
            "stage-owned artifact\n", encoding="utf-8"
        )
        return {
            "roc_auc": 0.75,
            "catalog_fraction": (
                context.items_with_sids.height / context.prepared.catalog_size
            ),
        }


class FakeCandgenEvaluator(IEvaluationStage):
    @property
    def name(self) -> str:
        return "candgen"

    def evaluate(self, context: EvaluationContext) -> dict[str, float]:
        return {"recall": 0.5, "ndcg": 0.25}


class MissingSidColumns(ISidPipeline):
    def train(self, df: pl.DataFrame) -> None:
        pass

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


def sample_embeddings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": [30, 20, 10, 999],
            "embed": [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [9.0, 9.0]],
        }
    )


class SidRunnerTest(unittest.TestCase):
    def _write_inputs(self, directory: Path) -> DatasetFiles:
        interactions_path = directory / "interactions.parquet"
        embeddings_path = directory / "embeddings.parquet"
        interactions = pl.concat([sample_interactions()] * 3)
        interactions.write_parquet(interactions_path)
        sample_embeddings().write_parquet(embeddings_path)
        return DatasetFiles(
            size="50m",
            interactions_path=interactions_path,
            embeddings_path=embeddings_path,
        )

    @staticmethod
    def _stage_resolver(name: str) -> IEvaluationStage:
        factories = {
            "ranking": FakeRankingEvaluator,
            "candgen": FakeCandgenEvaluator,
        }
        return factories[name]()

    def test_sid_and_both_common_evaluation_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            dataset = self._write_inputs(directory)
            result = run_sid_experiment(
                SidExperimentConfig(
                    dataset=dataset,
                    sid_spec="tests.test_sid_runner:FakeSid",
                    enabled_stages=("ranking", "candgen"),
                    reports_dir=directory / "reports",
                    core_min_interactions_per_item=2,
                    test_days=1,
                ),
                stage_resolver=self._stage_resolver,
            )

            self.assertEqual(result["sid"]["columns"], ["sid_0", "sid_1"])
            self.assertEqual(result["sid"]["item_count"], 3)
            statistics = result["sid"]["statistics"]
            self.assertIsNone(statistics["path_average_similarity"])
            self.assertEqual(
                statistics["codebook_quality"][
                    "codebook_utilization_rate_at_1_layer"
                ],
                1.0,
            )
            self.assertEqual(set(result["evaluations"]), {"ranking", "candgen"})
            self.assertEqual(result["evaluations"]["ranking"]["roc_auc"], 0.75)
            self.assertEqual(result["evaluations"]["candgen"]["recall"], 0.5)
            run_dir = Path(result["artifacts"]["directory"])
            self.assertTrue((run_dir / "ranking" / "curve.txt").is_file())
            semantic_ids = pl.read_parquet(result["artifacts"]["semantic_ids"])
            self.assertEqual(
                semantic_ids.columns,
                ["item_id", "catalog_item_idx", "sid_0", "sid_1"],
            )
            self.assertEqual(semantic_ids["item_id"].to_list(), [10, 20, 30])
            self.assertEqual(semantic_ids["catalog_item_idx"].to_list(), [0, 1, 2])

    def test_cli_uses_only_shared_dataset_and_report_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            dataset = self._write_inputs(directory)
            reports_dir = directory / "reports"

            with (
                patch(
                    "sids_rank_vs_generate.cli.resolve_dataset",
                    return_value=dataset,
                ),
                patch.dict(
                    DEFAULT_STAGE_FACTORIES,
                    {
                        "ranking": FakeRankingEvaluator,
                        "candgen": FakeCandgenEvaluator,
                    },
                    clear=True,
                ),
            ):
                exit_code = main(
                    [
                        "--dataset-size",
                        "50m",
                        "--sid",
                        "tests.test_sid_runner:FakeSid",
                        "--ranking",
                        "--candgen",
                        "--reports-dir",
                        str(reports_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            metrics_paths = list(
                (reports_dir / "50m" / "FakeSid").glob("*/metrics.json")
            )
            self.assertEqual(len(metrics_paths), 1)
            metrics_path = metrics_paths[0]
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment"], "sid_evaluation")
            self.assertEqual(set(payload["evaluations"]), {"ranking", "candgen"})
            self.assertFalse(list(reports_dir.rglob("*.md")))

    def test_sid_output_requires_integer_semantic_id_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            dataset = self._write_inputs(directory)
            with self.assertRaisesRegex(ValueError, "at least one sid_<level>"):
                run_sid_experiment(
                    SidExperimentConfig(
                        dataset=dataset,
                        sid_spec="tests.test_sid_runner:MissingSidColumns",
                        reports_dir=directory / "reports",
                        core_min_interactions_per_item=2,
                        test_days=1,
                    )
                )

    def test_yambda_50m_prepares_the_shared_embedding_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            source = directory / "source"
            source.mkdir()
            inputs = self._write_inputs(source)

            dataset = resolve_dataset(
                "50m",
                cache_root=directory / "cache",
                source_embeddings_path=inputs.embeddings_path,
                downloader=lambda *_args, **_kwargs: str(
                    inputs.interactions_path
                ),
            )

            self.assertTrue(dataset.interactions_path.is_file())
            self.assertTrue(dataset.embeddings_path.is_file())
            self.assertEqual(dataset.size, "50m")
            prepared_embeddings = pl.read_parquet(dataset.embeddings_path)
            self.assertEqual(
                prepared_embeddings.columns,
                ["item_id", "normalized_embed"],
            )

    def test_sid_is_required_by_the_common_runner(self) -> None:
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
