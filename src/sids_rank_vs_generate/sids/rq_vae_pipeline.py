import polars as pl

from dataclasses import asdict
from .base import ISidPipeline
from .rq_vae import RQVAE
from experiments.Config_rqvae_rqkmeans import RQVAEConfig


class RQVAEPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config or RQVAEConfig()
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        dim = len(df["normalized_embed"][0])
        self.model = RQVAE(dim=dim, **asdict(self.config))
        self.model.fit(df)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)