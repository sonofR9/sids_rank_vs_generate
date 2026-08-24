from dataclasses import asdict

import polars as pl
from dataclasses import dataclass
from .base import ISidPipeline
from .balance_rq_kmeans import BalanceRQKMeans
from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything

@dataclass(frozen=True)
class BalanceRQKMeansConfig:
    codebook_sizes: tuple[int, ...] = (256, 256, 256)
    kmeans_batch_size: int = 65_536
    predict_batch_size: int = 16_384
    max_iter: int = 50
    device: str = "auto"
    seed: int = 42

class BalanceRQKMeansPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        seed_everything(SID_EXPERIMENT_SEED)
        parameters = asdict(self.config)
        parameters.pop("seed", None)
        if parameters.get("codebook_sizes") is not None:
            parameters.pop("num_levels", None)
            parameters.pop("codebook_size", None)
        self.model = BalanceRQKMeans(**parameters)
        self.model.fit(df, tracker=self.tracker)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)
