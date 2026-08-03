from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_INVALID_MLFLOW_KEY_CHARACTER = re.compile(r"[^a-zA-Z0-9_./: \-]")


@dataclass(frozen=True)
class MLflowConfig:
    """Optional MLflow tracking configuration for a complete experiment."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "sids-rank-vs-generate"
    run_name: str | None = None
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("MLflow experiment_name must not be empty")
        if self.run_name is not None and not self.run_name.strip():
            raise ValueError("MLflow run_name must not be empty")
        if any(not key.strip() for key, _value in self.tags):
            raise ValueError("MLflow tag names must not be empty")


def _parameter_value(value: Any) -> str | int | float | bool:
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _mlflow_key(value: Any) -> str:
    key = str(value).replace("@", "_at_")
    key = _INVALID_MLFLOW_KEY_CHARACTER.sub("_", key).strip()
    if not key:
        raise ValueError("MLflow parameter and metric names must not be empty")
    return key


def flatten_parameters(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    """Flatten nested configuration mappings into MLflow parameter keys."""

    flattened: dict[str, str | int | float | bool] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_parameters(value, prefix=name))
        else:
            flattened[name] = _parameter_value(value)
    return flattened


def flatten_metrics(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Extract finite numeric leaves from a nested result mapping."""

    flattened: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, prefix=name))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            flattened[name] = float(value)
    return flattened


class ExperimentTracker:
    """Small tracking boundary shared by custom PyTorch training loops."""

    @contextmanager
    def start_run(
        self,
        run_name: str,
        *,
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[ExperimentTracker]:
        del run_name, nested, tags
        yield self

    def log_params(self, params: Mapping[str, Any]) -> None:
        del params

    def log_metrics(
        self,
        metrics: Mapping[str, int | float],
        *,
        step: int | None = None,
    ) -> None:
        del metrics, step

    def log_artifacts(self, directory: Path, *, artifact_path: str | None = None) -> None:
        del directory, artifact_path

    def info(self) -> dict[str, Any]:
        return {"enabled": False}


class NullTracker(ExperimentTracker):
    """Default tracker used when MLflow is disabled."""


class MlflowTracker(ExperimentTracker):
    """Thin, lazily imported adapter over MLflow's fluent tracking API."""

    def __init__(self, config: MLflowConfig, *, mlflow_module: Any | None = None):
        if not config.enabled:
            raise ValueError("MlflowTracker requires MLflowConfig(enabled=True)")
        if mlflow_module is None:
            try:
                mlflow_module = importlib.import_module("mlflow")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "MLflow tracking is enabled but mlflow is not installed; "
                    "install the project with the 'mlflow' extra"
                ) from error
        self._mlflow = mlflow_module
        self._config = config
        if config.tracking_uri is not None:
            self._mlflow.set_tracking_uri(config.tracking_uri)
        self._mlflow.set_experiment(config.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str,
        *,
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[ExperimentTracker]:
        with self._mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=dict(tags or {}),
        ):
            yield self

    def log_params(self, params: Mapping[str, Any]) -> None:
        if params:
            self._mlflow.log_params(
                {_mlflow_key(key): _parameter_value(value) for key, value in params.items()}
            )

    def log_metrics(
        self,
        metrics: Mapping[str, int | float],
        *,
        step: int | None = None,
    ) -> None:
        finite_metrics = {
            _mlflow_key(key): float(value)
            for key, value in metrics.items()
            if not isinstance(value, bool) and math.isfinite(float(value))
        }
        if finite_metrics:
            self._mlflow.log_metrics(finite_metrics, step=step)

    def log_artifacts(self, directory: Path, *, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def info(self) -> dict[str, Any]:
        active_run = self._mlflow.active_run()
        return {
            "enabled": True,
            "experiment_name": self._config.experiment_name,
            "run_id": active_run.info.run_id if active_run is not None else None,
        }


def create_tracker(config: MLflowConfig) -> ExperimentTracker:
    if config.enabled:
        return MlflowTracker(config)
    return NullTracker()
