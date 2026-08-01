from dataclasses import asdict

import polars as pl
from .base import ISidPipeline
from .rq_kmeans import RQKMeans

class RQKMeansPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        self.model = RQKMeans(**asdict(self.config))
        self.model.fit(df)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)