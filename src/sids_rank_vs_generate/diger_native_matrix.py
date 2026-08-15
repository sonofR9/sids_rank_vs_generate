"""Сравнение режимов декодирования native DIGER"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .tracking import MLflowConfig, create_tracker, flatten_parameters


@dataclass(frozen=True)
class DecodeCandidate:
    """Один режим native декодирования"""

    name: str
    num_beams: int
    constrain_to_catalog: bool


DEFAULT_CANDIDATES = (
    DecodeCandidate("native-b150", 150, False),
    DecodeCandidate("catalog-b150", 150, True),
    DecodeCandidate("catalog-b300", 300, True),
)


def select_candidate(
    results: dict[str, dict[str, Any]],
    candidates: Sequence[DecodeCandidate] = DEFAULT_CANDIDATES,
) -> DecodeCandidate:
    """Выбрать режим только по validation"""

    missing = [candidate.name for candidate in candidates if candidate.name not in results]
    if missing:
        raise ValueError(f"missing DIGER validation results: {missing}")

    def key(candidate: DecodeCandidate) -> tuple[float, float, int]:
        metrics = results[candidate.name]["metrics"]
        return (
            float(metrics["item_recall@100"]),
            float(metrics["item_mrr@100"]),
            -candidate.num_beams,
        )

    return max(candidates, key=key)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_result(
    path: Path,
    *,
    checkpoint: Path,
    split: str,
    candidate: DecodeCandidate,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint": str(checkpoint),
        "split": split,
        "num_beams": candidate.num_beams,
        "constrain_to_catalog": candidate.constrain_to_catalog,
    }
    actual = {name: result.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"stale DIGER evaluation result at {path}: {actual}")
    if "metrics" not in result or "item_recall@100" not in result["metrics"]:
        raise ValueError(f"incomplete DIGER evaluation result at {path}")
    return result


def _run_evaluation(
    *,
    repo: Path,
    diger_python: Path,
    source: Path,
    checkpoint: Path,
    config: Path,
    output: Path,
    log_path: Path,
    split: str,
    candidate: DecodeCandidate,
    gpu: int,
    eval_batch_size: int,
    num_workers: int,
    max_examples: int | None,
) -> dict[str, Any]:
    cached = _load_result(
        output,
        checkpoint=checkpoint,
        split=split,
        candidate=candidate,
    )
    if cached is not None:
        print(f"Reuse {output}", flush=True)
        return cached

    command = [
        str(diger_python),
        "-u",
        "-m",
        "sids_rank_vs_generate.diger_native_evaluation",
        "--source",
        str(source),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--split",
        split,
        "--device",
        "cuda:0",
        "--eval-batch-size",
        str(eval_batch_size),
        "--num-workers",
        str(num_workers),
        "--num-beams",
        str(candidate.num_beams),
        "--cutoffs",
        "1,5,10,50,100",
        "--output",
        str(output),
    ]
    if candidate.constrain_to_catalog:
        command.append("--constrain-to-catalog")
    if max_examples is not None:
        command.extend(["--max-examples", str(max_examples)])

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    existing_pythonpath = environment.get("PYTHONPATH")
    source_path = str(repo / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Start split={split} candidate={candidate.name} output={output}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    result = _load_result(
        output,
        checkpoint=checkpoint,
        split=split,
        candidate=candidate,
    )
    assert result is not None
    return result


def _resolve_native_run(pilot_root: Path, variant: str) -> dict[str, Path]:
    completion_files = sorted((pilot_root / variant / "runs").glob("*/run_complete.json"))
    if len(completion_files) != 1:
        raise ValueError(
            f"expected one completed DIGER pilot for {variant}, found {len(completion_files)}"
        )
    completion_path = completion_files[0]
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "FINISHED":
        raise ValueError(f"DIGER pilot is not finished: {completion_path}")
    code_path = Path(completion["artifacts"]["joint_code_table"])
    suffix = ".code.json"
    if not code_path.name.endswith(suffix):
        raise ValueError(f"unexpected DIGER code table name: {code_path}")
    checkpoint = code_path.with_name(code_path.name.removesuffix(suffix) + ".pt")
    run_dir = completion_path.parent
    paths = {
        "completion": completion_path,
        "source": pilot_root / variant / "reference-source",
        "checkpoint": checkpoint,
        "config": run_dir / "diger_config.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing native DIGER artifacts: " + ", ".join(missing))
    return paths


def run_matrix(
    *,
    repo: Path,
    diger_python: Path,
    pilot_root: Path,
    output_root: Path,
    variant: str,
    gpu: int,
    eval_batch_size: int,
    num_workers: int,
    max_validation_examples: int,
    tracking: MLflowConfig,
) -> dict[str, Any]:
    """Подобрать decode на validation и один раз проверить test"""

    paths = _resolve_native_run(pilot_root, variant)
    variant_root = output_root / variant
    validation_results: dict[str, dict[str, Any]] = {}
    for candidate in DEFAULT_CANDIDATES:
        validation_results[candidate.name] = _run_evaluation(
            repo=repo,
            diger_python=diger_python,
            source=paths["source"],
            checkpoint=paths["checkpoint"],
            config=paths["config"],
            output=variant_root / f"validation-{candidate.name}.json",
            log_path=variant_root / "logs" / f"validation-{candidate.name}.log",
            split="valid",
            candidate=candidate,
            gpu=gpu,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            max_examples=max_validation_examples,
        )
    selected = select_candidate(validation_results)
    selected_test = _run_evaluation(
        repo=repo,
        diger_python=diger_python,
        source=paths["source"],
        checkpoint=paths["checkpoint"],
        config=paths["config"],
        output=variant_root / f"test-{selected.name}.json",
        log_path=variant_root / "logs" / f"test-{selected.name}.log",
        split="test",
        candidate=selected,
        gpu=gpu,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_examples=None,
    )
    native = DEFAULT_CANDIDATES[0]
    native_test = (
        selected_test
        if selected == native
        else _run_evaluation(
            repo=repo,
            diger_python=diger_python,
            source=paths["source"],
            checkpoint=paths["checkpoint"],
            config=paths["config"],
            output=variant_root / f"test-{native.name}.json",
            log_path=variant_root / "logs" / f"test-{native.name}.log",
            split="test",
            candidate=native,
            gpu=gpu,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            max_examples=None,
        )
    )
    summary = {
        "schema_version": 1,
        "variant": variant,
        "checkpoint": str(paths["checkpoint"]),
        "max_validation_examples": max_validation_examples,
        "validation": validation_results,
        "selected_candidate": asdict(selected),
        "selected_test": selected_test,
        "native_test": native_test,
    }
    summary_path = variant_root / "summary.json"
    _atomic_json(summary_path, summary)

    tracker = create_tracker(tracking)
    with tracker.start_run(
        tracking.run_name or f"diger-native-{variant}",
        tags={
            "method": "diger-native",
            "variant": variant,
            "selection_split": "validation",
        },
    ):
        tracker.log_params(
            flatten_parameters(
                {
                    "variant": variant,
                    "checkpoint": str(paths["checkpoint"]),
                    "max_validation_examples": max_validation_examples,
                    "candidates": [asdict(candidate) for candidate in DEFAULT_CANDIDATES],
                    "selected_candidate": asdict(selected),
                }
            )
        )
        tracker.log_metrics(
            {
                f"selected.test.{name}": value
                for name, value in selected_test["metrics"].items()
            }
        )
        tracker.log_metrics(
            {
                f"native.test.{name}": value
                for name, value in native_test["metrics"].items()
            }
        )
        tracker.log_artifact(summary_path, artifact_path="diger_native")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--diger-python", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("both", "frqud"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-validation-examples", type=int, default=10_000)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="diger-native-v0.1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu < 0 or args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid native DIGER matrix settings")
    if args.max_validation_examples <= 0:
        raise ValueError("max_validation_examples must be positive")
    tracking = MLflowConfig(
        enabled=args.mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=f"diger-native-{args.variant}-pilot-checkpoints",
        tags=(("series", "diger-native-v0.1"), ("variant", args.variant)),
    )
    summary = run_matrix(
        repo=args.repo.expanduser().resolve(),
        diger_python=args.diger_python.expanduser().resolve(),
        pilot_root=args.pilot_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        variant=args.variant,
        gpu=args.gpu,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_validation_examples=args.max_validation_examples,
        tracking=tracking,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
