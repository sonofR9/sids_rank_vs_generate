"""Environment-driven TransAct-only sweep used by the all-methods figure.

Required environment variables:
    SID_METHOD: one of the method keys in ``build_pipeline``
    NUM_LEVELS: SID depth
    CODEBOOK_SIZE: nominal codebook size

The local SID cache prevents equivalent variants (for example RQVAE, item
LogQ, and pure-SID LogQ) from retraining the same deterministic SID model.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.evaluation.ranking_stage import RankingConfig
from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    TrainingConfig,
    YambdaDatasetConfig,
)
from sids_rank_vs_generate.sids import (
    ISidPipeline,
    RQKMeansConfig,
    RQKMeansPipeline,
    RQVAEConfig,
    RQVAEPipeline,
)
from sids_rank_vs_generate.sids.balance_rq_kmeans_pipeline import (
    BalanceRQKMeansConfig,
    BalanceRQKMeansPipeline,
)
from sids_rank_vs_generate.sids.collision_resolution import (
    add_collision_resolving_sid,
)
from sids_rank_vs_generate.sids.config import PlumRQVAEConfig
from sids_rank_vs_generate.sids.plum_rq_vae_pipeline import PlumRQVAEPipeline
from sids_rank_vs_generate.sids.r3vae import R3VAEConfig, R3VAESid
from sids_rank_vs_generate.tracking import MLflowConfig


METHOD = os.environ["SID_METHOD"].strip().lower()
NUM_LEVELS = int(os.environ["NUM_LEVELS"])
CODEBOOK_SIZE = int(os.environ["CODEBOOK_SIZE"])
SEED = 42
CACHE_ROOT = Path(
    os.environ.get(
        "TRANSACT_SID_CACHE",
        "reports/transact_sid_token_history/sid_cache",
    )
)


METHOD_LABELS = {
    "r3vae_rqkmeans": "R3VAE (rqkmeans)",
    "r3vae_spherical": "R3VAE (spherical)",
    "rqkmeans": "RQKMeans",
    "rqvae_ecm": "RQVAE (ECM)",
    "rqvae": "RQVAE",
    "rqvae_rrs": "RQVAE (RRS)",
    "rqvae_token": "RQVAE-token",
    "plum": "PLUM",
    "balance": "BalanceRQKMeans",
    "rqvae_item_logq": "RQVAE + item LogQ",
    "rqvae_sid_logq": "RQVAE + pure SID LogQ",
}


class CollisionTokenPipeline(ISidPipeline):
    """Append the deterministic conflict token used by RQVAE-token."""

    def __init__(self, base: ISidPipeline, codebook_size: int) -> None:
        self.base = base
        self.config = base.config
        self.codebook_size = codebook_size

    def set_tracker(self, tracker) -> None:
        super().set_tracker(tracker)
        self.base.set_tracker(tracker)

    def train(self, frame: pl.DataFrame) -> None:
        self.base.train(frame)

    def update(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_collision_resolving_sid(
            self.base.update(frame),
            codebook_size=self.codebook_size,
        )


class CachedSidPipeline(ISidPipeline):
    """Serialize deterministic SID construction once per cache key."""

    def __init__(self, base: ISidPipeline, path: Path) -> None:
        self.base = base
        self.path = path
        self.config = base.config
        self._lock_file = None

    def set_tracker(self, tracker) -> None:
        super().set_tracker(tracker)
        self.base.set_tracker(tracker)

    def train(self, frame: pl.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock_file = lock_path.open("a+")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        if not self.path.is_file():
            self.base.train(frame)

    def update(self, frame: pl.DataFrame) -> pl.DataFrame:
        try:
            if self.path.is_file():
                result = pl.read_parquet(self.path)
            else:
                result = self.base.update(frame)
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                result.write_parquet(temporary)
                temporary.replace(self.path)
            return result
        finally:
            if self._lock_file is not None:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
                self._lock_file = None


def rqvae(assignment: str = "nearest") -> RQVAEPipeline:
    return RQVAEPipeline(
        RQVAEConfig(
            hidden_dim=256,
            latent_dim=128,
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
            batch_size=1_024,
            lr=5e-4,
            pretrain_epochs=3,
            max_epochs=20,
            assignment_method=assignment,
            psi_top_k=min(4, CODEBOOK_SIZE),
            device="cuda",
        )
    )


def build_pipeline() -> tuple[ISidPipeline, str]:
    if METHOD == "rqkmeans":
        return (
            RQKMeansPipeline(
                RQKMeansConfig(
                    num_levels=NUM_LEVELS,
                    codebook_size=CODEBOOK_SIZE,
                )
            ),
            f"rqkmeans-l{NUM_LEVELS}-c{CODEBOOK_SIZE}",
        )
    if METHOD in {"rqvae", "rqvae_item_logq", "rqvae_sid_logq"}:
        return rqvae(), f"rqvae-nearest-l{NUM_LEVELS}-c{CODEBOOK_SIZE}"
    if METHOD == "rqvae_rrs":
        return rqvae("rrs"), f"rqvae-rrs-l{NUM_LEVELS}-c{CODEBOOK_SIZE}"
    if METHOD == "rqvae_ecm":
        return rqvae("ecm"), f"rqvae-ecm-l{NUM_LEVELS}-c{CODEBOOK_SIZE}"
    if METHOD == "rqvae_token":
        return (
            CollisionTokenPipeline(rqvae(), CODEBOOK_SIZE),
            f"rqvae-nearest-token-l{NUM_LEVELS}-c{CODEBOOK_SIZE}",
        )
    if METHOD in {"r3vae_rqkmeans", "r3vae_spherical"}:
        initialization = METHOD.removeprefix("r3vae_")
        return (
            R3VAESid(
                R3VAEConfig(
                    codebook_sizes=(CODEBOOK_SIZE,) * NUM_LEVELS,
                    epochs=20,
                    batch_size=1_024,
                    learning_rate=5e-4,
                    kmeans_init_method=initialization,
                    seed=SEED,
                    device="cuda",
                )
            ),
            f"r3vae-{initialization}-l{NUM_LEVELS}-c{CODEBOOK_SIZE}",
        )
    if METHOD == "plum":
        sizes = (CODEBOOK_SIZE * 2, CODEBOOK_SIZE, CODEBOOK_SIZE // 2)
        if NUM_LEVELS != 3:
            raise ValueError("the plotted PLUM sweep contains only L=3")
        return (
            PlumRQVAEPipeline(
                PlumRQVAEConfig(
                    num_levels=NUM_LEVELS,
                    codebook_size=sizes[0],
                    codebook_sizes=sizes,
                    device="cuda",
                    seed=SEED,
                )
            ),
            "plum-cbs" + "x".join(map(str, sizes)),
        )
    if METHOD == "balance":
        if NUM_LEVELS != 3:
            raise ValueError("the plotted BalanceRQKMeans sweep contains only L=3")
        return (
            BalanceRQKMeansPipeline(
                BalanceRQKMeansConfig(
                    codebook_sizes=(CODEBOOK_SIZE,) * NUM_LEVELS,
                    device="cuda",
                    seed=SEED,
                )
            ),
            f"balance-l{NUM_LEVELS}-c{CODEBOOK_SIZE}",
        )
    raise ValueError(f"unknown SID_METHOD: {METHOD}")


pipeline, cache_key = build_pipeline()
pipeline = CachedSidPipeline(pipeline, CACHE_ROOT / f"{cache_key}.parquet")
run_name = (
    f"transact-sid-tokens-{METHOD}-l{NUM_LEVELS}-c{CODEBOOK_SIZE}-full-seed{SEED}"
)

EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=pipeline,
    training=TrainingConfig(
        ranking=RankingConfig(enabled_models=("transact",)),
    ),
    tracking=MLflowConfig(
        run_name=run_name,
        tags=(
            ("evaluation", "transact_sid_level_tokens"),
            ("method", METHOD_LABELS[METHOD]),
            ("method_key", METHOD),
            ("num_levels", str(NUM_LEVELS)),
            ("codebook_size", str(CODEBOOK_SIZE)),
            ("seed", str(SEED)),
            ("dataset", "50m"),
            ("data_scope", "full"),
        ),
    ),
    reports_dir=Path("reports/transact_sid_token_history/runs"),
)
