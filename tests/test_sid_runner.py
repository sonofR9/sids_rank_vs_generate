from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.cli import main
from sids_rank_vs_generate.evaluation import EvaluationContext, IEvaluationStage
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
    def _write_inputs(self, directory: Path) -> tuple[Path, Path]:
        interactions_path = directory / "interactions.parquet"
        embeddings_path = directory / "embeddings.parquet"
        sample_interactions().write_parquet(interactions_path)
        sample_embeddings().write_parquet(embeddings_path)
        return interactions_path, embeddings_path

    def test_sid_and_both_evaluation_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path, embeddings_path = self._write_inputs(Path(tmpdir))
            result = run_sid_experiment(
                SidExperimentConfig(
                    interactions_path=interactions_path,
                    embeddings_path=embeddings_path,
                    sid_spec="tests.test_sid_runner:FakeSid",
                    evaluator_specs={
                        "ranking": (
                            "tests.test_sid_runner:FakeRankingEvaluator"
                        ),
                        "candgen": (
                            "tests.test_sid_runner:FakeCandgenEvaluator"
                        ),
                    },
                    core_min_interactions_per_item=2,
                    test_days=1,
                    seed=7,
                )
            )

            self.assertEqual(result["sid"]["columns"], ["sid_0", "sid_1"])
            self.assertEqual(result["sid"]["item_count"], 3)
            self.assertEqual(set(result["evaluations"]), {"ranking", "candgen"})
            self.assertEqual(result["evaluations"]["ranking"]["roc_auc"], 0.75)
            self.assertEqual(result["evaluations"]["candgen"]["recall"], 0.5)

    def test_cli_writes_full_metrics_and_human_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            interactions_path, embeddings_path = self._write_inputs(directory)
            metrics_path = directory / "metrics.json"
            report_path = directory / "report.md"

            exit_code = main(
                [
                    "--interactions",
                    str(interactions_path),
                    "--embeddings",
                    str(embeddings_path),
                    "--sid",
                    "tests.test_sid_runner:FakeSid",
                    "--ranking-evaluator",
                    "tests.test_sid_runner:FakeRankingEvaluator",
                    "--candgen-evaluator",
                    "tests.test_sid_runner:FakeCandgenEvaluator",
                    "--core-min-interactions",
                    "2",
                    "--test-days",
                    "1",
                    "--seed",
                    "7",
                    "--metrics-json",
                    str(metrics_path),
                    "--report-md",
                    str(report_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment"], "sid_evaluation")
            self.assertEqual(set(payload["evaluations"]), {"ranking", "candgen"})
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# SID experiment: FakeSid", report)
            self.assertIn("### ranking", report)
            self.assertIn("### candgen", report)

    def test_sid_output_requires_integer_semantic_id_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            interactions_path, embeddings_path = self._write_inputs(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "at least one sid_<level>"):
                run_sid_experiment(
                    SidExperimentConfig(
                        interactions_path=interactions_path,
                        embeddings_path=embeddings_path,
                        sid_spec="tests.test_sid_runner:MissingSidColumns",
                        evaluator_specs={},
                        core_min_interactions_per_item=2,
                        test_days=1,
                    )
                )

    def test_cli_uses_the_default_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            interactions_path, embeddings_path = self._write_inputs(directory)
            reports_dir = directory / "reports"

            exit_code = main(
                [
                    "--interactions",
                    str(interactions_path),
                    "--embeddings",
                    str(embeddings_path),
                    "--sid",
                    "tests.test_sid_runner:FakeSid",
                    "--core-min-interactions",
                    "2",
                    "--test-days",
                    "1",
                    "--reports-dir",
                    str(reports_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((reports_dir / "FakeSid.json").is_file())
            self.assertTrue((reports_dir / "FakeSid.md").is_file())


if __name__ == "__main__":
    unittest.main()
