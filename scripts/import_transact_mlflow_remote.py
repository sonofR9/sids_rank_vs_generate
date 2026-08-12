#!/usr/bin/env python3
"""Import the offline TransAct JSON bundle into the shared MLflow server."""

from __future__ import annotations

import json
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


REMOTE_URI = "http://51.250.47.231:5000"
EXPERIMENT_NAME = "transact-sid-token-history-20260812"
REPOSITORY = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPOSITORY / "reports/transact_sid_token_history/mlflow_export"
MANIFEST_PATH = EXPORT_DIR / "manifest.json"
SOURCE_RUN_TAG = "migration.source_run_id"
DATA_COMPLETE_TAG = "migration.json_data_complete"
ARTIFACTS_COMPLETE_TAG = "migration.artifacts_complete"


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_or_create_experiment(client: MlflowClient):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            tags={"migration.format": "offline-json-v1"},
        )
        experiment = client.get_experiment(experiment_id)
    return experiment


def existing_runs(client: MlflowClient, experiment_id: str):
    result = {}
    for run in client.search_runs([experiment_id], max_results=10_000):
        source_id = run.data.tags.get(SOURCE_RUN_TAG)
        if source_id:
            result[source_id] = run
    return result


def create_run(client, experiment_id, source_id, payload, mapping):
    tags = {
        key: value
        for key, value in payload["tags"].items()
        if not key.startswith("mlflow.")
    }
    tags[SOURCE_RUN_TAG] = source_id
    tags["migration.source_tracking_uri"] = payload["source"]["tracking_uri"]
    source_parent = payload["run"]["parent_source_run_id"]
    if source_parent is not None:
        tags["mlflow.parentRunId"] = mapping[source_parent]
    return client.create_run(
        experiment_id,
        start_time=payload["run"]["start_time"],
        tags=tags,
        run_name=payload["run"]["name"],
    )


def import_data(client, destination_run_id, payload):
    params = [Param(key, value) for key, value in payload["params"].items()]
    for batch in chunks(params, 100):
        client.log_batch(destination_run_id, params=batch)

    tags = [
        RunTag(key, value)
        for key, value in payload["tags"].items()
        if not key.startswith("mlflow.")
    ]
    for batch in chunks(tags, 100):
        client.log_batch(destination_run_id, tags=batch)

    metrics = []
    for key, history in payload["metric_histories"].items():
        metrics.extend(
            Metric(
                key=key,
                value=point["value"],
                timestamp=point["timestamp"],
                step=point["step"],
            )
            for point in history
        )
    for batch in chunks(metrics, 1_000):
        client.log_batch(destination_run_id, metrics=batch)
    client.set_tag(destination_run_id, DATA_COMPLETE_TAG, "true")


def import_artifacts(client, destination_run_id, payload):
    for relative_path in payload["mlflow_artifacts"]:
        local_path = EXPORT_DIR / relative_path
        artifact_relative = local_path.relative_to(
            EXPORT_DIR / "mlflow_artifacts" / payload["source"]["run_id"]
        )
        parent = str(artifact_relative.parent)
        client.log_artifact(
            destination_run_id,
            str(local_path),
            artifact_path=None if parent == "." else parent,
        )
    local_directory = payload["local_run_directory"]
    if local_directory is not None:
        client.log_artifacts(
            destination_run_id,
            str(REPOSITORY / local_directory),
            artifact_path="run_output",
        )
    client.set_tag(destination_run_id, ARTIFACTS_COMPLETE_TAG, "true")


def main() -> int:
    manifest = load_json(MANIFEST_PATH)
    if manifest["experiment_name"] != EXPERIMENT_NAME:
        raise ValueError("export manifest has an unexpected experiment name")
    client = MlflowClient(tracking_uri=REMOTE_URI)
    client.search_experiments(max_results=1)
    experiment = get_or_create_experiment(client)
    existing = existing_runs(client, experiment.experiment_id)
    mapping = {source_id: run.info.run_id for source_id, run in existing.items()}

    for index, item in enumerate(manifest["runs"], start=1):
        source_id = item["source_run_id"]
        payload = load_json(EXPORT_DIR / item["json"])
        destination_run = existing.get(source_id)
        if destination_run is None:
            destination_run = create_run(
                client,
                experiment.experiment_id,
                source_id,
                payload,
                mapping,
            )
            existing[source_id] = destination_run
        destination_id = destination_run.info.run_id
        mapping[source_id] = destination_id

        current = client.get_run(destination_id)
        if current.data.tags.get(DATA_COMPLETE_TAG) != "true":
            import_data(client, destination_id, payload)
        current = client.get_run(destination_id)
        if current.data.tags.get(ARTIFACTS_COMPLETE_TAG) != "true":
            import_artifacts(client, destination_id, payload)
        if payload["run"]["status"] != "RUNNING":
            client.set_terminated(
                destination_id,
                status=payload["run"]["status"],
                end_time=payload["run"]["end_time"],
            )
        print(
            f"IMPORT {index}/{manifest['run_count']} source={source_id} "
            f"destination={destination_id} name={payload['run']['name']}",
            flush=True,
        )

    print(
        f"COMPLETE experiment={EXPERIMENT_NAME} runs={manifest['run_count']} "
        f"destination={REMOTE_URI}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
