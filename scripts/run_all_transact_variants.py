#!/usr/bin/env python3
"""Run the exact SID variants shown in the all-methods Recall@100 figure."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


METHOD_KEYS = {
    "R3VAE (rqkmeans)": "r3vae_rqkmeans",
    "R3VAE (spherical)": "r3vae_spherical",
    "RQKMeans": "rqkmeans",
    "RQVAE (ECM)": "rqvae_ecm",
    "RQVAE": "rqvae",
    "RQVAE (RRS)": "rqvae_rrs",
    "RQVAE-token": "rqvae_token",
    "PLUM": "plum",
    "BalanceRQKMeans": "balance",
    "RQVAE + item LogQ": "rqvae_item_logq",
    "RQVAE + pure SID LogQ": "rqvae_sid_logq",
}


def plotted_variants(path: Path) -> list[tuple[str, int, int]]:
    frame = pd.read_csv(path)
    selected = pd.concat(
        [
            frame[
                frame["codebook_size"].eq(256)
                & frame["num_levels"].isin((2, 3, 4))
            ],
            frame[
                frame["num_levels"].eq(3)
                & frame["codebook_size"].isin((128, 256, 512, 1024))
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(["method", "num_levels", "codebook_size"])
    unknown = set(selected["method"]) - set(METHOD_KEYS)
    if unknown:
        raise ValueError(f"unknown plotted methods: {sorted(unknown)}")
    variants = [
        (METHOD_KEYS[row.method], int(row.num_levels), int(row.codebook_size))
        for row in selected.itertuples(index=False)
    ]
    return sorted(variants)


def run_name(method: str, levels: int, size: int) -> str:
    return f"transact-sid-tokens-{method}-l{levels}-c{size}-full-seed42"


def finished_run_names(tracking_uri: str, experiment_name: str) -> set[str]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return set()
    runs = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="pandas",
        max_results=10_000,
    )
    column = "tags.mlflow.runName"
    return set(runs[column].dropna()) if column in runs else set()


def launch(
    variant: tuple[str, int, int],
    gpu: str,
    args: argparse.Namespace,
) -> tuple[subprocess.Popen, object, Path, tuple[str, int, int]]:
    method, levels, size = variant
    name = run_name(method, levels, size)
    log_path = args.log_dir / f"{name}.log"
    log_file = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "SID_METHOD": method,
            "NUM_LEVELS": str(levels),
            "CODEBOOK_SIZE": str(size),
            "TRANSACT_SID_CACHE": str(args.sid_cache),
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        sys.executable,
        "run.py",
        "experiments/transact_all_variants.py",
        "--skip-generation",
        "--mlflow",
        "--mlflow-tracking-uri",
        args.tracking_uri,
        "--mlflow-experiment-name",
        args.experiment_name,
        "--mlflow-run-name",
        name,
    ]
    process = subprocess.Popen(
        command,
        cwd=args.repository,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"START gpu={gpu} pid={process.pid} {name}", flush=True)
    return process, log_file, log_path, variant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        type=Path,
        default=Path(
            "reports/figures/all_methods_sid_quality/generation_points.csv"
        ),
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--experiment-name",
        default="transact-sid-token-history-20260812",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("reports/transact_sid_token_history/logs"),
    )
    parser.add_argument(
        "--sid-cache",
        type=Path,
        default=Path("reports/transact_sid_token_history/sid_cache"),
    )
    parser.add_argument("--rerun-finished", action="store_true")
    args = parser.parse_args()
    args.repository = Path(__file__).resolve().parents[1]
    args.points = (args.repository / args.points).resolve()
    args.log_dir = (args.repository / args.log_dir).resolve()
    args.sid_cache = (args.repository / args.sid_cache).resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.sid_cache.mkdir(parents=True, exist_ok=True)

    variants = plotted_variants(args.points)
    finished = (
        set()
        if args.rerun_finished
        else finished_run_names(args.tracking_uri, args.experiment_name)
    )
    pending = [
        variant
        for variant in variants
        if run_name(*variant) not in finished
    ]
    print(
        f"MATRIX total={len(variants)} finished={len(variants) - len(pending)} "
        f"pending={len(pending)} experiment={args.experiment_name}",
        flush=True,
    )
    if not pending:
        return 0

    available_gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not available_gpus:
        raise ValueError("--gpus must contain at least one GPU")
    active: dict[subprocess.Popen, tuple[object, Path, tuple[str, int, int], str]] = {}
    failures = []
    while pending or active:
        while pending and available_gpus:
            gpu = available_gpus.pop(0)
            process, log_file, log_path, variant = launch(
                pending.pop(0), gpu, args
            )
            active[process] = (log_file, log_path, variant, gpu)
        if not active:
            break
        done = [process for process in active if process.poll() is not None]
        if not done:
            time.sleep(2)
            continue
        for process in done:
            log_file, log_path, variant, gpu = active.pop(process)
            return_code = process.wait()
            log_file.close()
            available_gpus.append(gpu)
            name = run_name(*variant)
            if return_code:
                failures.append((name, return_code, log_path))
                print(
                    f"FAIL gpu={gpu} exit={return_code} {name} log={log_path}",
                    flush=True,
                )
            else:
                print(f"DONE gpu={gpu} {name}", flush=True)

    if failures:
        print(f"FAILED {len(failures)} run(s):", flush=True)
        for name, return_code, log_path in failures:
            print(f"  {name}: exit={return_code} log={log_path}", flush=True)
        return 1
    print(f"COMPLETE {len(variants)} TransAct runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
