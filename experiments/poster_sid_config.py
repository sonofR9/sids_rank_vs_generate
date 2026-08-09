"""Environment-driven config for the poster Recall@100/ICR matrix."""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import (
    R3VAEConfig,
    R3VAESid,
    RQKMeansConfig,
    RQKMeansPipeline,
    RQVAEConfig,
    RQVAEPipeline,
)
from sids_rank_vs_generate.tracking import MLflowConfig


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


SID_MODEL = _choice("SID_MODEL", "rqkmeans", {"rqkmeans", "rqvae", "r3vae"})
R3_INIT = _choice("R3_INIT", "spherical", {"spherical", "rqkmeans"})
ASSIGNMENT_METHOD = _choice(
    "ASSIGNMENT_METHOD",
    "nearest",
    {"nearest", "rrs", "ecm"},
)
COLLISION_RESOLUTION = _choice(
    "COLLISION_RESOLUTION",
    "none",
    {"none", "token"},
)
CODEBOOK_SIZE = _positive_int("CODEBOOK_SIZE", 256)
NUM_LEVELS = _positive_int("NUM_LEVELS", 3)
DATASET_SIZE = os.environ.get("DATASET_SIZE", "50m").strip()
MODEL_CACHE_DIR = Path(
    os.environ.get("MODEL_CACHE_DIR", "reports/poster_cache")
)
SEED = 42

if SID_MODEL != "rqvae" and ASSIGNMENT_METHOD != "nearest":
    raise ValueError("ASSIGNMENT_METHOD applies only to RQVAE")
if ASSIGNMENT_METHOD in {"rrs", "ecm"} and COLLISION_RESOLUTION == "token":
    raise ValueError("RRS/ECM and collision token are mutually exclusive")

cache_stem = f"{DATASET_SIZE}-full-l{NUM_LEVELS}-c{CODEBOOK_SIZE}-seed{SEED}"
if SID_MODEL == "rqkmeans":
    sid_pipeline = RQKMeansPipeline(
        RQKMeansConfig(
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
        )
    )
    init_method = "rqkmeans"
elif SID_MODEL == "rqvae":
    sid_pipeline = RQVAEPipeline(
        RQVAEConfig(
            hidden_dim=256,
            latent_dim=128,
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
            batch_size=1_024,
            lr=5e-4,
            pretrain_epochs=3,
            max_epochs=20,
            assignment_method=ASSIGNMENT_METHOD,
            psi_top_k=min(4, CODEBOOK_SIZE),
            checkpoint_path=(
                MODEL_CACHE_DIR / f"rqvae-{cache_stem}.pt"
            ),
        )
    )
    init_method = "rqkmeans"
else:
    sid_pipeline = R3VAESid(
        R3VAEConfig(
            codebook_sizes=(CODEBOOK_SIZE,) * NUM_LEVELS,
            epochs=20,
            batch_size=1_024,
            learning_rate=5e-4,
            kmeans_init_method=R3_INIT,
            seed=SEED,
            device="auto",
            checkpoint_path=(
                MODEL_CACHE_DIR / f"r3vae-{R3_INIT}-{cache_stem}.pt"
            ),
        )
    )
    init_method = R3_INIT

run_name = (
    f"poster-{SID_MODEL}-{init_method}-{ASSIGNMENT_METHOD}-"
    f"{COLLISION_RESOLUTION}-l{NUM_LEVELS}-c{CODEBOOK_SIZE}-full-seed{SEED}"
)

EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size=DATASET_SIZE),
    sid_pipeline=sid_pipeline,
    training=TrainingConfig(
        generation=GenerationConfig(
            resolve_collisions=COLLISION_RESOLUTION == "token",
        )
    ),
    tracking=MLflowConfig(
        run_name=run_name,
        tags=(
            ("poster", "true"),
            ("sid_model", SID_MODEL),
            ("init", init_method),
            ("assignment", ASSIGNMENT_METHOD),
            ("collision_resolution", COLLISION_RESOLUTION),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("num_levels", str(NUM_LEVELS)),
            ("seed", str(SEED)),
            ("dataset", DATASET_SIZE),
            ("data_scope", "full"),
        ),
    ),
    reports_dir=Path("reports/poster_runs"),
)
