import polars as pl

from .bpe import BPE
from .rq_vae_pipeline import RQVAEPipeline
from .config import RQVAEConfig 
from dataclasses import dataclass, fields

@dataclass(frozen=True)
class BPE_RQVAEConfig(RQVAEConfig):
    bpe_num_merges: int = 1000
    bpe_min_frequency: int = 5


class BPERQVAEPipeline(RQVAEPipeline):
    def __init__(self, config: BPE_RQVAEConfig):
        self.bpe = BPE(
            num_merges=config.bpe_num_merges,
            min_frequency=config.bpe_min_frequency,
        )
        super().__init__(RQVAEConfig(**{
            field.name: getattr(config, field.name)
            for field in fields(RQVAEConfig)
        }))


    def train(self, df: pl.DataFrame) -> None:
        super().train(df)

        sids = self.model.transform(df)

        sid_columns = [
            f"sid_{i}"
            for i in range(self.model.num_levels)
        ]

        self.bpe.fit_polars(
            sids,
            sid_columns=sid_columns,
        )

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        sids = self.model.transform(df)

        sid_columns = [
            f"sid_{i}"
            for i in range(self.model.num_levels)
        ]

        return self.bpe.transform_polars(
            sids,
            sid_columns=sid_columns,
        )
