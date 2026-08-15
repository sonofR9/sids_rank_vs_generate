from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

from .data.preprocessing import PreparedInteractions
from .data.protocols import THREE_WAY_PROTOCOL_ID, validate_protocol_id
from .datasets import DatasetFiles, resolve_dataset
from .pipeline import PipelineConfig, run_pipeline
from .reproducibility import SID_EXPERIMENT_SEED, seed_everything
from .tracking import MLflowConfig, create_tracker, flatten_parameters


DIGER_REPOSITORY = "https://github.com/junchen-fu/DIGER.git"
DIGER_COMMIT = "cd72dcdc25d28e0acfa5aea14788ad863b41ab4f"
DIGER_DATASET_NAME = "yambda50m"
_TEST_RESULTS_RE = re.compile(r"Test Results:\s*(\{.*\})")
_BEST_VALIDATION_RE = re.compile(r"Best Validation Score:\s*([^\s]+)")
_NUMPY_FLOAT_RE = re.compile(
    r"np\.float(?:16|32|64)?\(" r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)" r"\)"
)
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class DigerReferenceConfig:
    """Настройки запуска зафиксированной реализации DIGER"""

    variant: str = "frqud"
    dataset_size: str = "50m"
    protocol_id: str = THREE_WAY_PROTOCOL_ID
    core_min_interactions_per_item: int = 5
    history_days: float | None = None
    test_days: float = 7.0
    timestamp_unit_seconds: float = 5.0
    validation_fraction: float = 0.1
    catalog_from_fit: bool = False
    latest_interactions: int | None = None
    min_history_len: int = 1
    max_history_len: int = 50
    max_train_examples: int | None = None
    seed: int = SID_EXPERIMENT_SEED
    num_levels: int = 3
    codebook_size: int = 256
    rqvae_epochs: int = 2_000
    joint_epochs: int = 120
    train_batch_size: int = 256
    eval_batch_size: int = 32
    num_workers: int = 8
    num_beams: int = 20
    early_stop: int = 15

    def __post_init__(self) -> None:
        validate_protocol_id(self.protocol_id)
        if self.variant not in {"frqud", "sdud", "both"}:
            raise ValueError("variant must be 'frqud', 'sdud', or 'both'")
        positive = {
            "core_min_interactions_per_item": self.core_min_interactions_per_item,
            "test_days": self.test_days,
            "timestamp_unit_seconds": self.timestamp_unit_seconds,
            "min_history_len": self.min_history_len,
            "max_history_len": self.max_history_len,
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "rqvae_epochs": self.rqvae_epochs,
            "joint_epochs": self.joint_epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "num_beams": self.num_beams,
            "early_stop": self.early_stop,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive DIGER settings required: {invalid}")
        if self.max_history_len < self.min_history_len:
            raise ValueError("max_history_len must be at least min_history_len")
        if self.num_beams < 10:
            raise ValueError("num_beams must be at least 10 for upstream evaluation")
        if self.latest_interactions is not None and self.latest_interactions <= 0:
            raise ValueError("latest_interactions must be positive")
        if self.max_train_examples is not None and self.max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive")


@dataclass(frozen=True)
class DigerReferencePaths:
    root: Path
    source: Path
    prepared: Path
    rqvae: Path
    run: Path

    @classmethod
    def build(cls, root: Path, run_name: str) -> "DigerReferencePaths":
        if not _RUN_NAME_RE.fullmatch(run_name):
            raise ValueError(
                "run_name may contain only letters, digits, '.', '_' and '-'"
            )
        resolved = root.expanduser().resolve()
        return cls(
            root=resolved,
            source=resolved / "reference-source",
            prepared=resolved / "prepared" / DIGER_DATASET_NAME,
            rqvae=resolved / "rqvae" / DIGER_DATASET_NAME,
            run=resolved / "runs" / run_name,
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_column(frame: pl.DataFrame) -> str:
    for column in ("normalized_embed", "embed"):
        if column in frame.columns:
            return column
    raise ValueError("item embeddings must contain normalized_embed or embed")


def _embedding_matrix(frame: pl.DataFrame, column: str) -> np.ndarray:
    if frame.is_empty():
        raise ValueError("item embeddings must not be empty")
    first = frame[column][0]
    if first is None or len(first) == 0:
        raise ValueError("item embedding vectors must not be empty")
    width = len(first)
    dtype = frame.schema[column]
    if isinstance(dtype, pl.Array):
        values = frame[column].to_numpy()
    else:
        values = frame[column].list.to_array(width).to_numpy()
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (frame.height, width):
        raise ValueError("item embeddings must form a dense two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("item embeddings contain non-finite values")
    return matrix


def _mapped_item_features(
    raw_features: pl.DataFrame,
    prepared: PreparedInteractions,
) -> pl.DataFrame:
    if "item_id" not in raw_features.columns:
        raise ValueError("item embeddings must contain item_id")
    if raw_features["item_id"].n_unique() != raw_features.height:
        raise ValueError("item embeddings must contain one row per item_id")
    feature_columns = [column for column in raw_features.columns if column != "item_id"]
    mapping = prepared.item_mapping.rename({"item_id": "mapped_item_id"})
    mapped = (
        raw_features.rename({"item_id": "original_item_id"})
        .join(mapping, on="original_item_id", how="inner")
        .select("mapped_item_id", *feature_columns)
        .rename({"mapped_item_id": "item_id"})
        .sort("item_id")
    )
    if mapped.height != prepared.catalog_size:
        raise ValueError(
            "item embeddings do not cover the prepared DIGER catalog: "
            f"{mapped.height} of {prepared.catalog_size}"
        )
    expected = list(range(prepared.catalog_size))
    if mapped["item_id"].to_list() != expected:
        raise ValueError("prepared DIGER item ids must be dense and ordered")
    return mapped


def build_reference_examples(
    prepared: PreparedInteractions,
    *,
    min_history_len: int,
    max_history_len: int,
) -> dict[str, list[dict[str, Any]]]:
    """Преобразовать общее разбиение в JSONL авторов DIGER"""

    if min_history_len <= 0 or max_history_len < min_history_len:
        raise ValueError("invalid DIGER history lengths")
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    history: list[str] = []
    current_uid: Any = object()
    for row in prepared.data.sort(["uid", "timestamp"]).iter_rows(named=True):
        if row["uid"] != current_uid:
            current_uid = row["uid"]
            history = []
        token = f"item_{int(row['item_id'])}"
        if len(history) >= min_history_len:
            example = {
                "inter_history": history[-max_history_len:],
                "target_id": token,
            }
            timestamp = row["timestamp"]
            if timestamp >= prepared.test_start_timestamp:
                splits["test"].append(example)
            elif timestamp >= prepared.validation_start_timestamp:
                splits["valid"].append(example)
            else:
                splits["train"].append(example)
        history.append(token)
    if any(not rows for rows in splits.values()):
        counts = {name: len(rows) for name, rows in splits.items()}
        raise ValueError(f"DIGER split contains no examples: {counts}")
    return splits


def _limit_train_examples(
    examples: dict[str, list[dict[str, Any]]],
    maximum: int | None,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if maximum is None or len(examples["train"]) <= maximum:
        return examples
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(examples["train"])), maximum))
    return {
        **examples,
        "train": [examples["train"][index] for index in indices],
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    temporary.replace(path)


def prepare_reference_data(
    dataset: DatasetFiles,
    destination: Path,
    config: DigerReferenceConfig,
) -> dict[str, Any]:
    seed_everything(config.seed)
    raw_features = pl.read_parquet(dataset.embeddings_path)
    prepared = run_pipeline(
        PipelineConfig(
            interactions_path=dataset.interactions_path,
            core_min_interactions_per_item=config.core_min_interactions_per_item,
            history_days=config.history_days,
            test_days=config.test_days,
            timestamp_unit_seconds=config.timestamp_unit_seconds,
            validation_fraction=config.validation_fraction,
            catalog_from_fit=config.catalog_from_fit,
            latest_interactions=config.latest_interactions,
            protocol_id=config.protocol_id,
        ),
        eligible_items=raw_features.select("item_id"),
    )
    item_features = _mapped_item_features(raw_features, prepared)
    embedding_column = _embedding_column(item_features)
    embeddings = _embedding_matrix(item_features, embedding_column)
    examples = _limit_train_examples(
        build_reference_examples(
            prepared,
            min_history_len=config.min_history_len,
            max_history_len=config.max_history_len,
        ),
        config.max_train_examples,
        config.seed,
    )

    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination / DIGER_DATASET_NAME
    embedding_path = destination / "Yambda50m.emb.npy"
    np.save(embedding_path, embeddings, allow_pickle=False)
    embedding_sha256 = _file_sha256(embedding_path)
    item_map = {"[PAD]": 0}
    item_map.update(
        {f"item_{item_id}": item_id + 1 for item_id in range(prepared.catalog_size)}
    )
    _atomic_json(prefix.with_suffix(".emb_map.json"), item_map)
    for split, rows in examples.items():
        _write_jsonl(prefix.with_suffix(f".{split}.jsonl"), rows)
    prepared.item_mapping.write_parquet(destination / "item_mapping.parquet")

    manifest = {
        "schema_version": 1,
        "reference_repository": DIGER_REPOSITORY,
        "reference_commit": DIGER_COMMIT,
        "dataset": dataset.size,
        "protocol": prepared.summary(),
        "embedding_column": embedding_column,
        "embedding_shape": list(embeddings.shape),
        "embedding_sha256": embedding_sha256,
        "split_examples": {name: len(rows) for name, rows in examples.items()},
        "config": asdict(config),
        "paths": {
            "embedding": str(embedding_path),
            "item_map": str(prefix.with_suffix(".emb_map.json")),
            "item_mapping": str(destination / "item_mapping.parquet"),
        },
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def checkout_reference(source: Path) -> None:
    if source.exists():
        if not (source / ".git").is_dir():
            raise RuntimeError(f"DIGER source path is not a git checkout: {source}")
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if head != DIGER_COMMIT:
            raise RuntimeError(
                f"DIGER checkout must stay pinned to {DIGER_COMMIT}, found {head}"
            )
        remote = subprocess.check_output(
            ["git", "-C", str(source), "remote", "get-url", "origin"],
            text=True,
        ).strip()
        if remote.rstrip("/") not in {
            DIGER_REPOSITORY.rstrip("/"),
            DIGER_REPOSITORY.removesuffix(".git").rstrip("/"),
        }:
            raise RuntimeError(f"unexpected DIGER origin: {remote}")
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    clone_environment = dict(os.environ)
    clone_environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            DIGER_REPOSITORY,
            str(source),
        ],
        check=True,
        env=clone_environment,
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", DIGER_COMMIT],
        check=True,
        env=clone_environment,
    )


def _variant_settings(variant: str) -> dict[str, Any]:
    if variant == "frqud":
        return {
            "balance_loss_weight": 0.0,
            "use_adaptive_selection": True,
            "hot_threshold_ratio": 1.5,
            "usage_momentum": 0.99,
            "use_learnable_sigma_gumbel": False,
            "gumbel_hard_switch_epoch": 0,
        }
    if variant == "sdud":
        return {
            "use_adaptive_selection": False,
            "use_learnable_sigma_gumbel": True,
            "use_plain_code_loss": False,
            "use_simple_uncertainty_loss": True,
            "lr_sigma": 0.001,
            "initial_std": 2.0,
            "noise_type": "gumbel",
            "sigma_lambda": 1.7,
            "gumbel_hard_switch_epoch": 0,
        }
    if variant == "both":
        return {
            "balance_loss_weight": 0.0,
            "use_adaptive_selection": True,
            "hot_threshold_ratio": 1.5,
            "usage_momentum": 0.99,
            "use_learnable_sigma_gumbel": True,
            "use_plain_code_loss": False,
            "lr_sigma": 0.001,
            "sigma_reg_weight": 2.0,
            "initial_std": 1.0,
            "noise_type": "gumbel",
            "use_cosine_annealing": False,
            "use_dynamic_sigma_lr": True,
            "gate_loss_weight": 0.0,
            "gumbel_hard_switch_epoch": 0,
        }
    raise ValueError(f"unsupported DIGER variant: {variant}")


def build_upstream_config(
    config: DigerReferenceConfig,
    *,
    prepared_dir: Path,
    run_dir: Path,
    rqvae_checkpoint: Path,
    embedding_dimension: int,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "dataset": DIGER_DATASET_NAME,
        "seed": config.seed,
        "reproducibility": True,
        "log_dir": str(run_dir / "logs"),
        "map_path": ".emb_map.json",
        "semantic_emb_path": "Yambda50m.emb.npy",
        "semantic_hidden_size": embedding_dimension,
        "code_num": config.codebook_size,
        "code_length": config.num_levels + 1,
        "epochs": config.joint_epochs,
        "lr_rec": 0.001,
        "weight_decay": 0.05,
        "joint_optimization": True,
        "end_to_end": True,
        "freeze_semantic_embedding": True,
        "use_indicator_ste": True,
        "freeze_id_encoder": True,
        "freeze_id_encoder_layers": 0,
        "freeze_id_decoder": True,
        "freeze_id_epochs": 0,
        "freeze_rq": False,
        "stop_gumbel_sampling_epoch": 0,
        "lr_id": 0.00001,
        "gumbel_tau": 2.0,
        "code_loss_weight": 1.0,
        "recon_loss_weight": 1.0,
        "vq_loss_weight": 1.0,
        "qs_loss_weight": 0.0,
        "balance_loss_weight": 0.0,
        "gate_loss_weight": 0.0,
        "kl_loss_weight": 0.0,
        "dec_cl_loss_weight": 0.0,
        "qs_beta": 0.25,
        "sim": "cos",
        "alpha": 1.0,
        "tau": 0.07,
        "loss_type": "mse",
        "warmup_steps": 500,
        "batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "num_workers": config.num_workers,
        "eval_step": 1,
        "learner": "AdamW",
        "data_path": str(prepared_dir.parent),
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 1,
        "reference_effective_batch_size": config.train_batch_size,
        "preserve_reference_forward_batch": True,
        "sync_quantizer_stats": True,
        "max_his_len": config.max_history_len,
        "early_stop": config.early_stop,
        "metrics": "recall@1,recall@5,ndcg@5,recall@10,ndcg@10",
        "valid_metric": "ndcg@10",
        # Collator авторов оставляет max_length - 1
        "max_length": config.max_history_len + 1,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "d_model": 128,
        "d_ff": 512,
        "num_heads": 4,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "activation_function": "relu",
        "feed_forward_proj": "relu",
        "num_beams": config.num_beams,
        "rqvae_path": str(rqvae_checkpoint),
        "num_emb_list": [config.codebook_size] * config.num_levels,
        "beta": 0.25,
        "e_dim": 256,
        "layers": [2048, 1024, 512],
        "vq_type": "vq",
        "dist": "l2",
        "kmeans_init": False,
        "kmeans_iters": 100,
        "dropout_prob": 0.0,
        "bn": False,
        "sk_epsilons": [0.003] * config.num_levels,
        "sk_iters": 50,
        "use_gaq": False,
        "gaq_gamma": 0.6,
        "gaq_eps": 1e-7,
        "use_soft_frequency": False,
        "use_gate_network": False,
    }
    values.update(_variant_settings(config.variant))
    return values


def _tee_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


def train_reference_rqvae(
    *,
    source: Path,
    prepared_dir: Path,
    destination: Path,
    diger_python: Path,
    gpu: int,
    config: DigerReferenceConfig,
    embedding_sha256: str,
) -> Path:
    canonical = destination / "best_collision_model.pth"
    manifest_path = destination / "rqvae_manifest.json"
    expected_manifest = {
        "schema_version": 1,
        "reference_commit": DIGER_COMMIT,
        "embedding_sha256": embedding_sha256,
        "num_levels": config.num_levels,
        "codebook_size": config.codebook_size,
        "epochs": config.rqvae_epochs,
    }
    if canonical.is_file():
        if not manifest_path.is_file():
            raise RuntimeError(
                "existing DIGER RQ-VAE has no manifest; use another work root"
            )
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                "existing DIGER RQ-VAE does not match this data/config; "
                "use another work root"
            )
        return canonical
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        str(diger_python),
        "-u",
        str(source / "scripts" / "rqvae" / "main.py"),
        "--lr",
        "1e-3",
        "--epochs",
        str(config.rqvae_epochs),
        "--batch_size",
        "1024",
        "--weight_decay",
        "1e-4",
        "--lr_scheduler_type",
        "linear",
        "--e_dim",
        "256",
        "--quant_loss_weight",
        "1.0",
        "--beta",
        "0.25",
        "--num_emb_list",
        *([str(config.codebook_size)] * config.num_levels),
        "--sk_epsilons",
        *(["0.003"] * config.num_levels),
        "--sk_iters",
        "50",
        "--layers",
        "2048",
        "1024",
        "512",
        "--vq_type",
        "vq",
        "--loss_type",
        "mse",
        "--dist",
        "l2",
        "--device",
        "cuda:0",
        "--kmeans_init",
        "True",
        "--data_path",
        str(prepared_dir / "Yambda50m.emb.npy"),
        "--ckpt_dir",
        str(destination / "training"),
    ]
    _tee_command(
        command,
        cwd=destination,
        environment=environment,
        log_path=destination / "rqvae.log",
    )
    candidates = sorted(
        (destination / "training").rglob("best_collision_model.pth"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError("upstream DIGER RQ-VAE did not produce a checkpoint")
    shutil.copy2(candidates[-1], canonical)
    _atomic_json(manifest_path, expected_manifest)
    return canonical


def parse_reference_metrics(log_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        test_match = _TEST_RESULTS_RE.search(line)
        if test_match:
            payload = _NUMPY_FLOAT_RE.sub(r"\1", test_match.group(1))
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, dict):
                metrics.update(
                    {f"test.{name}": float(value) for name, value in parsed.items()}
                )
        validation_match = _BEST_VALIDATION_RE.search(line)
        if validation_match:
            try:
                metrics["validation.best_score"] = float(validation_match.group(1))
            except ValueError:
                pass
    if not any(name.startswith("test.") for name in metrics):
        raise RuntimeError("could not find final DIGER test metrics in the run log")
    return metrics


def export_reference_sids(
    code_path: Path,
    *,
    catalog_size: int,
    item_mapping_path: Path,
    destination: Path,
    num_levels: int,
) -> dict[str, str]:
    codes = json.loads(code_path.read_text(encoding="utf-8"))
    if len(codes) == catalog_size + 1:
        codes = codes[1:]
    if len(codes) != catalog_size:
        raise ValueError(
            f"DIGER code table has {len(codes)} items, expected {catalog_size}"
        )
    if any(len(code) < num_levels for code in codes):
        raise ValueError("DIGER code table contains truncated semantic ids")
    raw = pl.DataFrame(
        {
            "item_id": range(catalog_size),
            **{
                f"sid_{level}": [int(code[level]) for code in codes]
                for level in range(num_levels)
            },
        }
    )
    with_conflict_levels = min(num_levels + 1, min(len(code) for code in codes))
    with_conflict = pl.DataFrame(
        {
            "item_id": range(catalog_size),
            **{
                f"sid_{level}": [int(code[level]) for code in codes]
                for level in range(with_conflict_levels)
            },
        }
    )
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "semantic_ids.parquet"
    conflict_path = destination / "semantic_ids_with_conflict.parquet"
    original_path = destination / "semantic_ids_original_items.parquet"
    raw.write_parquet(raw_path)
    with_conflict.write_parquet(conflict_path)
    (
        pl.read_parquet(item_mapping_path)
        .join(raw, on="item_id", how="inner")
        .rename({"original_item_id": "item_id", "item_id": "catalog_item_idx"})
        .select("item_id", "catalog_item_idx", *[f"sid_{i}" for i in range(num_levels)])
        .write_parquet(original_path)
    )
    return {
        "raw": str(raw_path),
        "with_conflict": str(conflict_path),
        "original_items": str(original_path),
    }


def _newest_code_table(run_dir: Path) -> Path:
    candidates = sorted(
        (run_dir / "myckpt" / DIGER_DATASET_NAME).rglob("*.code.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError("upstream DIGER run did not produce a SID code table")
    return candidates[-1]


def _finalize_reference_artifacts(
    *,
    config: DigerReferenceConfig,
    paths: DigerReferencePaths,
    data_manifest: dict[str, Any],
    rqvae_checkpoint: Path,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    log_path = paths.run / "diger.log"
    config_path = paths.run / "diger_config.json"
    if not log_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("completed DIGER log and config are required")
    embedding_shape = data_manifest["embedding_shape"]
    metrics = parse_reference_metrics(log_path)
    code_path = _newest_code_table(paths.run)
    sid_artifacts = export_reference_sids(
        code_path,
        catalog_size=int(embedding_shape[0]),
        item_mapping_path=paths.prepared / "item_mapping.parquet",
        destination=paths.run / "sid_mapping",
        num_levels=config.num_levels,
    )
    completion_path = paths.run / "run_complete.json"
    result = {
        "schema_version": 1,
        "status": "FINISHED",
        "reference_repository": DIGER_REPOSITORY,
        "reference_commit": DIGER_COMMIT,
        "config": asdict(config),
        "data": data_manifest,
        "metrics": metrics,
        "artifacts": {
            "rqvae_checkpoint": str(rqvae_checkpoint),
            "joint_code_table": str(code_path),
            "log": str(log_path),
            **sid_artifacts,
        },
    }
    _atomic_json(completion_path, result)
    artifacts = (
        completion_path,
        config_path,
        log_path,
        *map(Path, sid_artifacts.values()),
    )
    return result, artifacts


def finalize_existing_reference_run(
    *,
    config: DigerReferenceConfig,
    paths: DigerReferencePaths,
    tracking: MLflowConfig,
) -> dict[str, Any]:
    """Упаковать завершённый upstream run без повторного обучения"""

    completion_path = paths.run / "run_complete.json"
    if completion_path.exists():
        raise FileExistsError(f"DIGER run is already finalized: {completion_path}")
    manifest_path = paths.prepared / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepared DIGER manifest is missing: {manifest_path}")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("config") != asdict(config):
        raise ValueError("finalize config does not match the prepared DIGER run")
    rqvae_checkpoint = paths.rqvae / "best_collision_model.pth"
    if not rqvae_checkpoint.is_file():
        raise FileNotFoundError(
            f"DIGER RQ-VAE checkpoint is missing: {rqvae_checkpoint}"
        )

    tracker = create_tracker(tracking)
    with tracker.start_run(
        tracking.run_name or f"{paths.run.name}-finalize",
        tags={
            "method": "diger-reference",
            "variant": config.variant,
            "reference_commit": DIGER_COMMIT,
            "protocol_id": config.protocol_id,
            "recovered_without_retraining": "true",
        },
    ):
        tracker.log_params(
            flatten_parameters(
                {
                    "diger": asdict(config),
                    "reference": {
                        "repository": DIGER_REPOSITORY,
                        "commit": DIGER_COMMIT,
                    },
                    "data": data_manifest,
                    "recovered_without_retraining": True,
                }
            )
        )
        result, artifacts = _finalize_reference_artifacts(
            config=config,
            paths=paths,
            data_manifest=data_manifest,
            rqvae_checkpoint=rqvae_checkpoint,
        )
        tracker.log_metrics(result["metrics"])
        for artifact in artifacts:
            tracker.log_artifact(artifact, artifact_path="diger_reference_v0_1")
    return result


def run_reference_experiment(
    *,
    config: DigerReferenceConfig,
    paths: DigerReferencePaths,
    diger_env_bin: Path,
    gpu: int,
    tracking: MLflowConfig,
) -> dict[str, Any]:
    if gpu < 0:
        raise ValueError("gpu must be non-negative")
    diger_python = diger_env_bin / "python"
    accelerate = diger_env_bin / "accelerate"
    if not diger_python.is_file() or not accelerate.is_file():
        raise FileNotFoundError(
            "DIGER reference environment must provide python and accelerate: "
            f"{diger_env_bin}"
        )
    completion_path = paths.run / "run_complete.json"
    if completion_path.exists():
        raise FileExistsError(
            f"DIGER run already completed: {completion_path}; choose another run name"
        )

    paths.run.mkdir(parents=True, exist_ok=True)
    checkout_reference(paths.source)
    dataset = resolve_dataset(config.dataset_size)
    data_manifest = prepare_reference_data(dataset, paths.prepared, config)
    rqvae_checkpoint = train_reference_rqvae(
        source=paths.source,
        prepared_dir=paths.prepared,
        destination=paths.rqvae,
        diger_python=diger_python,
        gpu=gpu,
        config=config,
        embedding_sha256=str(data_manifest["embedding_sha256"]),
    )
    embedding_shape = data_manifest["embedding_shape"]
    upstream_config = build_upstream_config(
        config,
        prepared_dir=paths.prepared,
        run_dir=paths.run,
        rqvae_checkpoint=rqvae_checkpoint,
        embedding_dimension=int(embedding_shape[1]),
    )
    config_path = paths.run / "diger_config.json"
    _atomic_json(config_path, upstream_config)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = paths.run / "diger.log"

    tracker = create_tracker(tracking)
    with tracker.start_run(
        tracking.run_name or paths.run.name,
        tags={
            "method": "diger-reference",
            "variant": config.variant,
            "reference_commit": DIGER_COMMIT,
            "protocol_id": config.protocol_id,
        },
    ):
        tracker.log_params(
            flatten_parameters(
                {
                    "diger": asdict(config),
                    "reference": {
                        "repository": DIGER_REPOSITORY,
                        "commit": DIGER_COMMIT,
                    },
                    "data": data_manifest,
                    "gpu": gpu,
                }
            )
        )
        _tee_command(
            [
                str(accelerate),
                "launch",
                "--config_file",
                str(paths.source / "accelerate_config.yaml"),
                str(paths.source / "main.py"),
                "--config",
                str(config_path),
            ],
            cwd=paths.run,
            environment=environment,
            log_path=log_path,
        )
        result, artifacts = _finalize_reference_artifacts(
            config=config,
            paths=paths,
            data_manifest=data_manifest,
            rqvae_checkpoint=rqvae_checkpoint,
        )
        tracker.log_metrics(result["metrics"])
        for artifact in artifacts:
            tracker.log_artifact(artifact, artifact_path="diger_reference_v0_1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned upstream DIGER implementation on Yambda-50M."
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--diger-env-bin", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", choices=("frqud", "sdud", "both"), default="frqud")
    parser.add_argument("--protocol-id", default=THREE_WAY_PROTOCOL_ID)
    parser.add_argument("--latest-interactions", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--rqvae-epochs", type=int, default=2_000)
    parser.add_argument("--joint-epochs", type=int, default=120)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Package a completed upstream run without retraining it.",
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-name", default="sids-rank-vs-generate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_name = args.run_name or (
        f"diger-reference-v0-1-{args.variant}-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    config = DigerReferenceConfig(
        variant=args.variant,
        protocol_id=args.protocol_id,
        latest_interactions=args.latest_interactions,
        max_train_examples=args.max_train_examples,
        rqvae_epochs=args.rqvae_epochs,
        joint_epochs=args.joint_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        num_beams=args.num_beams,
    )
    paths = DigerReferencePaths.build(args.work_root, run_name)
    tracking = MLflowConfig(
        enabled=args.mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=(f"{run_name}-recovered" if args.finalize_existing else run_name),
        tags=(("method", "diger-reference-v0.1"), ("variant", args.variant)),
    )
    if args.finalize_existing:
        result = finalize_existing_reference_run(
            config=config,
            paths=paths,
            tracking=tracking,
        )
    else:
        result = run_reference_experiment(
            config=config,
            paths=paths,
            diger_env_bin=args.diger_env_bin.expanduser().resolve(),
            gpu=args.gpu,
            tracking=tracking,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
