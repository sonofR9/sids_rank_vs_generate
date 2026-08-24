from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import torch


REQUIRED_FILES = (
    "semantic_ids.parquet",
    "user_semantic_ids_fit.parquet",
    "generation/best_mapper.pt",
    "generation/best_model.pt",
    "generation/semantic_ids.parquet",
    "generation/user_semantic_ids_observed.parquet",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _find_result(results_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for metrics_path in results_root.rglob("metrics.json"):
        try:
            metrics = _load_json(metrics_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if metrics.get("tracking", {}).get("run_id") == run_id:
            matches.append((metrics_path.parent, metrics))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one result for MLflow run {run_id}, found {len(matches)}"
        )
    return matches[0]


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} must contain a checkpoint mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} does not contain model weights")
    fields = (
        "epoch",
        "best_epoch",
        "best_metric",
        "best_validation_loss",
        "best_validation_selection_value",
        "best_validation_top_k_accuracy",
        "validation_selection_metric",
        "epochs_trained",
    )
    return {
        "parameters": len(state),
        **{name: checkpoint[name] for name in fields if name in checkpoint},
    }


def verify_result(results_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    result_dir, metrics = _find_result(results_root, run_id)
    generation = metrics.get("evaluations", {}).get("generation", {})
    if not generation.get("metrics"):
        raise ValueError("final generation test metrics are missing")

    files: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_FILES:
        path = result_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required artifact is missing or empty: {path}")
        summary: dict[str, Any] = {"size_bytes": path.stat().st_size}
        if path.suffix == ".parquet":
            frame = pl.read_parquet(path)
            if frame.is_empty():
                raise ValueError(f"required parquet is empty: {path}")
            summary.update({"rows": frame.height, "columns": frame.columns})
        elif path.suffix == ".pt":
            summary.update(_checkpoint_summary(path))
        files[relative] = summary

    manifest = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verified_on_host": socket.gethostname(),
        "verified_by_uid": os.getuid(),
        "mlflow_run_id": run_id,
        "result_dir": str(result_dir),
        "files": files,
        "final_test_metrics_present": True,
    }
    return result_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    result_dir, manifest = verify_result(args.results_root, args.run_id)
    if args.write_manifest:
        manifest_path = result_dir / "verification_manifest.json"
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        temporary_path.replace(manifest_path)
    print(result_dir)


if __name__ == "__main__":
    main()
