#!/usr/bin/env python3
"""Export the local TransAct MLflow experiment to an offline JSON bundle."""

from __future__ import annotations

import json
from pathlib import Path

from mlflow.tracking import MlflowClient


SOURCE_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "transact-sid-token-history-20260812"
REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    REPOSITORY / "reports/transact_sid_token_history/mlflow_export"
)
LOCAL_RUNS_ROOT = (
    REPOSITORY / "reports/transact_sid_token_history/runs"
)


def local_run_directories() -> dict[str, Path]:
    result = {}
    for metrics_path in LOCAL_RUNS_ROOT.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            run_id = payload["tracking"]["run_id"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if run_id:
            result[str(run_id)] = metrics_path.parent
    return result


def artifact_paths(client: MlflowClient, run_id: str, path: str = ""):
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from artifact_paths(client, run_id, item.path)
        else:
            yield item.path


def export_mlflow_artifacts(
    client: MlflowClient,
    run_id: str,
    paths: list[str],
) -> list[str]:
    exported = []
    root = OUTPUT_DIR / "mlflow_artifacts" / run_id
    for artifact_path in paths:
        downloaded = Path(
            client.download_artifacts(run_id, artifact_path, str(root))
        )
        exported.append(str(downloaded.relative_to(OUTPUT_DIR)))
    return exported


def main() -> int:
    client = MlflowClient(tracking_uri=SOURCE_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"local MLflow experiment not found: {EXPERIMENT_NAME}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = OUTPUT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    local_directories = local_run_directories()
    runs = client.search_runs([experiment.experiment_id], max_results=10_000)
    runs = sorted(
        runs,
        key=lambda run: (
            run.data.tags.get("mlflow.parentRunId") is not None,
            run.info.start_time or 0,
            run.info.run_id,
        ),
    )

    manifest_runs = []
    for index, run in enumerate(runs, start=1):
        run_id = run.info.run_id
        metric_histories = {
            key: [
                {
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                    "step": metric.step,
                }
                for metric in client.get_metric_history(run_id, key)
            ]
            for key in sorted(run.data.metrics)
        }
        source_artifact_paths = list(artifact_paths(client, run_id))
        exported_artifacts = export_mlflow_artifacts(
            client,
            run_id,
            source_artifact_paths,
        )
        local_directory = local_directories.get(run_id)
        payload = {
            "schema_version": 1,
            "source": {
                "tracking_uri": SOURCE_URI,
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.name,
                "run_id": run_id,
            },
            "run": {
                "name": run.data.tags.get("mlflow.runName"),
                "parent_source_run_id": run.data.tags.get("mlflow.parentRunId"),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "lifecycle_stage": run.info.lifecycle_stage,
            },
            "params": dict(sorted(run.data.params.items())),
            "tags": dict(sorted(run.data.tags.items())),
            "metric_histories": metric_histories,
            "mlflow_artifacts": exported_artifacts,
            "local_run_directory": (
                str(local_directory.relative_to(REPOSITORY))
                if local_directory is not None
                else None
            ),
        }
        json_path = runs_dir / f"{run_id}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        manifest_runs.append(
            {
                "source_run_id": run_id,
                "json": str(json_path.relative_to(OUTPUT_DIR)),
                "name": payload["run"]["name"],
                "parent_source_run_id": payload["run"]["parent_source_run_id"],
                "status": payload["run"]["status"],
                "local_run_directory": payload["local_run_directory"],
            }
        )
        print(
            f"EXPORT {index}/{len(runs)} run={run_id} "
            f"name={payload['run']['name']}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "source_tracking_uri": SOURCE_URI,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment.experiment_id,
        "run_count": len(manifest_runs),
        "parent_run_count": sum(
            item["parent_source_run_id"] is None for item in manifest_runs
        ),
        "child_run_count": sum(
            item["parent_source_run_id"] is not None for item in manifest_runs
        ),
        "runs": manifest_runs,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE runs={len(runs)} manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
