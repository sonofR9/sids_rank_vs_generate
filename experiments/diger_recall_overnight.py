"""Две ночные очереди масштабирования Recall@100"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RecallStage:
    sid_method: str
    sid_variant: str
    model_variant: str
    protocol: str
    sid_path: Path | None = None

    @property
    def name(self) -> str:
        return "-".join(
            (
                self.sid_variant,
                self.model_variant,
                self.protocol,
            )
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_lane_stages(
    lane: int,
    *,
    both_sid_path: Path,
    frqud_sid_path: Path,
) -> tuple[RecallStage, ...]:
    if lane == 0:
        return (
            RecallStage("diger", "both", "strong128", "poster", both_sid_path),
            RecallStage("diger", "both", "strong128", "three-way", both_sid_path),
            RecallStage("rqkmeans", "rqkmeans", "strong128", "poster"),
            RecallStage("rqkmeans", "rqkmeans", "strong128", "three-way"),
        )
    if lane == 1:
        return (
            RecallStage("diger", "both", "large256", "poster", both_sid_path),
            RecallStage("diger", "both", "large256", "three-way", both_sid_path),
            RecallStage("diger", "frqud", "strong128", "poster", frqud_sid_path),
            RecallStage(
                "diger",
                "frqud",
                "strong128",
                "three-way",
                frqud_sid_path,
            ),
        )
    raise ValueError("lane must be 0 or 1")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def _run_stage(
    stage: RecallStage,
    *,
    index: int,
    lane: int,
    gpu: int,
    repo: Path,
    state_dir: Path,
    reports_dir: Path,
    mlflow_uri: str,
) -> None:
    marker = state_dir / "completed" / f"{index:02d}-{stage.name}.json"
    fingerprint = stage.fingerprint()
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError(f"completion marker differs for {stage.name}")
        return

    log_path = state_dir / "logs" / f"{index:02d}-{stage.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(state_dir / "current-run", stage.name + "\n")
    _atomic_text(state_dir / "current-log.path", str(log_path) + "\n")
    _atomic_text(state_dir / "status", "RUNNING\n")

    command = [
        sys.executable,
        "-u",
        "run.py",
        "experiments/diger_recall_scale.py",
        "--skip-ranking",
        "--mlflow",
        "--mlflow-tracking-uri",
        mlflow_uri,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(repo / "src"),
            "DIGER_SID_METHOD": stage.sid_method,
            "DIGER_VARIANT": stage.sid_variant,
            "DIGER_RECALL_MODEL": stage.model_variant,
            "DIGER_RECALL_PROTOCOL": stage.protocol,
            "SID_REPORTS_DIR": str(reports_dir),
        }
    )
    if stage.sid_path is not None:
        environment["DIGER_SID_PATH"] = str(stage.sid_path)
    else:
        environment.pop("DIGER_SID_PATH", None)

    started_at = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        _atomic_text(state_dir / "status", "FAILED\n")
        raise RuntimeError(
            f"stage {stage.name} failed with code {completed.returncode}"
        )
    _atomic_json(
        marker,
        {
            "schema_version": 1,
            "lane": lane,
            "gpu": gpu,
            "stage": asdict(stage),
            "fingerprint": fingerprint,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "log": str(log_path),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--both-sid-path", type=Path, required=True)
    parser.add_argument("--frqud-sid-path", type=Path, required=True)
    parser.add_argument("--mlflow-uri", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.expanduser().resolve()
    state_dir = args.state_root.expanduser().resolve() / f"lane-{args.lane}"
    stages = build_lane_stages(
        args.lane,
        both_sid_path=args.both_sid_path.expanduser().resolve(),
        frqud_sid_path=args.frqud_sid_path.expanduser().resolve(),
    )
    for path in (repo / "run.py", args.both_sid_path, args.frqud_sid_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    state_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        state_dir / "queue.json",
        {
            "schema_version": 1,
            "lane": args.lane,
            "gpu": args.gpu,
            "stages": [asdict(stage) for stage in stages],
        },
    )
    try:
        for index, stage in enumerate(stages, start=1):
            _run_stage(
                stage,
                index=index,
                lane=args.lane,
                gpu=args.gpu,
                repo=repo,
                state_dir=state_dir,
                reports_dir=args.reports_dir.expanduser().resolve(),
                mlflow_uri=args.mlflow_uri,
            )
    except Exception:
        _atomic_text(state_dir / "status", "FAILED\n")
        raise
    _atomic_text(state_dir / "current-run", "none\n")
    _atomic_text(state_dir / "status", "FINISHED\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
