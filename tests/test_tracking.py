from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sids_rank_vs_generate.sids.rq_vae import RQVAE
from sids_rank_vs_generate.tracking import (
    ExperimentTracker,
    MLflowConfig,
    MlflowTracker,
    NullTracker,
    create_tracker,
    flatten_metrics,
    flatten_parameters,
)


class CaptureTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.metrics = []

    def log_metrics(self, metrics, *, step=None) -> None:
        self.metrics.append((step, dict(metrics)))


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.runs = []
        self.stack = []
        self.artifacts = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    @contextmanager
    def start_run(self, *, run_name, nested, tags):
        run = SimpleNamespace(
            info=SimpleNamespace(run_id=f"run-{len(self.runs) + 1}"),
            name=run_name,
            nested=nested,
            tags=tags,
            params={},
            metrics=[],
        )
        self.runs.append(run)
        self.stack.append(run)
        try:
            yield run
        finally:
            self.stack.pop()

    def active_run(self):
        return self.stack[-1] if self.stack else None

    def log_params(self, params) -> None:
        self.stack[-1].params.update(params)

    def log_metrics(self, metrics, *, step=None) -> None:
        self.stack[-1].metrics.append((step, metrics))

    def log_artifact(self, path) -> None:
        self.artifacts.append((self.stack[-1].name, path))


class TrackingTest(unittest.TestCase):
    def test_disabled_tracking_does_not_import_mlflow(self) -> None:
        with patch(
            "sids_rank_vs_generate.tracking.importlib.import_module"
        ) as import_module:
            tracker = create_tracker(MLflowConfig())

        self.assertIsInstance(tracker, NullTracker)
        import_module.assert_not_called()

    def test_flatten_helpers_keep_params_and_only_numeric_metrics(self) -> None:
        self.assertEqual(
            flatten_parameters(
                {"model": {"layers": (64, 32), "path": Path("model.pt")}}
            ),
            {
                "model.layers": "[64, 32]",
                "model.path": "model.pt",
            },
        )
        self.assertEqual(
            flatten_metrics(
                {
                    "train": {"loss": 1.5, "finished": True},
                    "artifact": "model.pt",
                }
            ),
            {"train.loss": 1.5},
        )

    def test_mlflow_tracker_creates_parent_and_child_runs(self) -> None:
        fake = FakeMlflow()
        config = MLflowConfig(
            enabled=True,
            tracking_uri="file:./mlruns",
            experiment_name="sid-tests",
        )
        tracker = MlflowTracker(config, mlflow_module=fake)

        with tracker.start_run("parent", tags={"pipeline": "rqvae"}):
            tracker.log_params({"seed": 42})
            with tracker.start_run("generation", nested=True) as child:
                child.log_metrics({"train.loss": 0.5}, step=1)
                child.log_artifact(Path("metrics.json"))
                self.assertEqual(child.info()["run_id"], "run-2")

        self.assertEqual(fake.tracking_uri, "file:./mlruns")
        self.assertEqual(fake.experiment_name, "sid-tests")
        self.assertEqual([run.name for run in fake.runs], ["parent", "generation"])
        self.assertFalse(fake.runs[0].nested)
        self.assertTrue(fake.runs[1].nested)
        self.assertEqual(fake.runs[0].params, {"seed": 42})
        self.assertEqual(fake.runs[1].metrics, [(1, {"train.loss": 0.5})])
        self.assertEqual(
            fake.artifacts,
            [("generation", "metrics.json")],
        )

    def test_mlflow_tracker_sanitizes_at_sign_in_metric_names(self) -> None:
        fake = FakeMlflow()
        tracker = MlflowTracker(
            MLflowConfig(enabled=True),
            mlflow_module=fake,
        )

        with tracker.start_run("generation"):
            tracker.log_metrics({"validation.recall@100": 0.25}, step=1)

        self.assertEqual(
            fake.runs[0].metrics,
            [(1, {"validation.recall_at_100": 0.25})],
        )

    def test_enabled_tracking_has_actionable_missing_dependency_error(self) -> None:
        with patch(
            "sids_rank_vs_generate.tracking.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ):
            with self.assertRaisesRegex(RuntimeError, "mlflow is not installed"):
                create_tracker(MLflowConfig(enabled=True))

    def test_rqvae_logs_train_and_validation_loss_components(self) -> None:
        tracker = CaptureTracker()
        model = RQVAE(
            dim=4,
            hidden_dim=4,
            latent_dim=4,
            num_levels=2,
            codebook_size=2,
            batch_size=4,
            num_workers=0,
            device="cpu",
            max_epochs=1,
            patience=1,
            val_part=0.25,
        )
        embeddings = np.arange(32, dtype=np.float32).reshape(8, 4) / 32.0

        model.fit(embeddings, tracker=tracker)

        metric_names = {
            name
            for _step, metrics in tracker.metrics
            for name in metrics
        }
        self.assertIn("train.reconstruction_loss", metric_names)
        self.assertIn("train.codebook_loss", metric_names)
        self.assertIn("train.commitment_loss", metric_names)
        self.assertIn("validation.loss", metric_names)


if __name__ == "__main__":
    unittest.main()
