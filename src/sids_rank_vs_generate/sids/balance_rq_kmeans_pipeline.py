from dataclasses import asdict

import polars as pl

from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything
from .base import ISidPipeline
from .balance_rq_kmeans import BalanceRQKMeans
from .config import BalanceRQKMeansConfig


def _model_parameters(config: BalanceRQKMeansConfig) -> dict:
    parameters = asdict(config)
    if parameters.get("codebook_sizes") is not None:
        parameters.pop("num_levels", None)
        parameters.pop("codebook_size", None)
    if parameters.get("device") == "auto":
        parameters["device"] = None
    return parameters


class BalanceRQKMeansPipeline(ISidPipeline):
    def __init__(self, config: BalanceRQKMeansConfig):
        self.config = config
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        seed_everything(SID_EXPERIMENT_SEED)
        self.model = BalanceRQKMeans(**_model_parameters(self.config))
        self.model.fit(df, tracker=self.tracker)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)
