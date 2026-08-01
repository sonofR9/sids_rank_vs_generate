from dataclasses import asdict

import polars as pl
from .base import ISidPipeline
from .rq_kmeans import RQKMeans

class RQKMeansPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        parameters = asdict(self.config)
        if parameters.get("codebook_sizes") is not None:
            parameters.pop("num_levels", None)
            parameters.pop("codebook_size", None)
        self.model = RQKMeans(**parameters)
        self.model.fit(df)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)
