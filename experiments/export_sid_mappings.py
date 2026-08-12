"""Environment-driven SID-only configurations for exporting item-ID mappings."""

from __future__ import annotations

import os
from pathlib import Path

from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline, RQVAEConfig, RQVAEPipeline
from sids_rank_vs_generate.sids.balance_rq_kmeans_pipeline import BalanceRQKMeansPipeline
from sids_rank_vs_generate.sids.config import BalanceRQKMeansConfig
from sids_rank_vs_generate.sids.config import PlumRQVAEConfig
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import PlumRQVAEPipeline
from sids_rank_vs_generate.sids.r3vae import R3VAEConfig, R3VAESid
from sids_rank_vs_generate.tracking import MLflowConfig


def positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


METHOD = os.environ.get("SID_METHOD", "rqkmeans").strip().lower()
LEVELS = positive_int("NUM_LEVELS", 3)
CODEBOOK_SIZE = positive_int("CODEBOOK_SIZE", 256)
ASSIGNMENT = os.environ.get("ASSIGNMENT", "nearest").strip().lower()
R3VAE_INIT = os.environ.get("R3VAE_INIT", "rqkmeans").strip().lower()
DATASET_SIZE = os.environ.get("DATASET_SIZE", "50m").strip()
REPORTS_DIR = Path(os.environ.get("SID_MAPPING_REPORTS_DIR", "reports/sid_mappings"))
INTERACTIONS_PATH = os.environ.get("SID_INTERACTIONS_PATH")
EMBEDDINGS_PATH = os.environ.get("SID_EMBEDDINGS_PATH")
RUN_NAME = os.environ.get(
    "MLFLOW_RUN_NAME",
    f"sid-mapping-{METHOD}-l{LEVELS}-c{CODEBOOK_SIZE}-{ASSIGNMENT}-seed42",
)

if METHOD == "rqkmeans":
    pipeline = RQKMeansPipeline(RQKMeansConfig(num_levels=LEVELS, codebook_size=CODEBOOK_SIZE))
elif METHOD == "rqvae":
    if ASSIGNMENT not in {"nearest", "rrs", "ecm"}:
        raise ValueError("ASSIGNMENT must be nearest, rrs, or ecm")
    pipeline = RQVAEPipeline(
        RQVAEConfig(
            hidden_dim=256,
            latent_dim=128,
            num_levels=LEVELS,
            codebook_size=CODEBOOK_SIZE,
            batch_size=1_024,
            lr=5e-4,
            pretrain_epochs=3,
            max_epochs=20,
            assignment_method=ASSIGNMENT,
            psi_top_k=min(4, CODEBOOK_SIZE),
            device=None,
        )
    )
elif METHOD == "r3vae":
    if ASSIGNMENT != "nearest":
        raise ValueError("R3VAE only supports ASSIGNMENT=nearest")
    if R3VAE_INIT not in {"rqkmeans", "spherical"}:
        raise ValueError("R3VAE_INIT must be rqkmeans or spherical")
    pipeline = R3VAESid(
        R3VAEConfig(
            codebook_sizes=(CODEBOOK_SIZE,) * LEVELS,
            epochs=20,
            batch_size=1_024,
            learning_rate=5e-4,
            kmeans_init_method=R3VAE_INIT,
            seed=42,
            device="auto",
        )
    )
elif METHOD == "balance":
    if ASSIGNMENT != "nearest":
        raise ValueError("BalanceRQKMeans only supports ASSIGNMENT=nearest")
    pipeline = BalanceRQKMeansPipeline(
        BalanceRQKMeansConfig(
            num_levels=LEVELS,
            codebook_size=CODEBOOK_SIZE,
            device=None,
        )
    )
elif METHOD == "plum":
    if ASSIGNMENT != "nearest":
        raise ValueError("PLUM only supports ASSIGNMENT=nearest")
    pipeline = PlumRQVAEPipeline(
        PlumRQVAEConfig(
            num_levels=LEVELS,
            codebook_sizes=tuple(
                (CODEBOOK_SIZE * 2) // (2**level) for level in range(LEVELS)
            ),
            batch_size=1_024,
            num_epochs=50,
            device="auto",
        )
    )
else:
    raise ValueError("SID_METHOD must be rqkmeans, rqvae, r3vae, balance, or plum")


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        size=DATASET_SIZE,
        interactions_path=Path(INTERACTIONS_PATH) if INTERACTIONS_PATH else None,
        embeddings_path=Path(EMBEDDINGS_PATH) if EMBEDDINGS_PATH else None,
    ),
    sid_pipeline=pipeline,
    tracking=MLflowConfig(
        run_name=RUN_NAME,
        tags=(
            ("sid_mapping_export", "true"),
            ("sid_model", METHOD),
            ("assignment", ASSIGNMENT),
            ("num_levels", str(LEVELS)),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("seed", "42"),
            ("dataset", DATASET_SIZE),
        ),
    ),
    reports_dir=REPORTS_DIR,
)
