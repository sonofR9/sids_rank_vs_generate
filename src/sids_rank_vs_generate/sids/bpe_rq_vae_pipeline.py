from __future__ import annotations

from dataclasses import dataclass, fields

import polars as pl

from .bpe import BPE
from .config import RQVAEConfig
from .rq_vae_pipeline import RQVAEPipeline


@dataclass(frozen=True)
class BPE_RQVAEConfig(RQVAEConfig):
    bpe_num_merges: int = 1000
    bpe_min_frequency: int = 5


class BPERQVAEPipeline(RQVAEPipeline):
    def __init__(
        self,
        config: BPE_RQVAEConfig,
    ):
        # Сохраняем полный BPE-конфиг отдельно,
        # потому что RQVAEPipeline ожидает чистый RQVAEConfig.
        self.bpe_config = config

        self.bpe = BPE(
            num_merges=config.bpe_num_merges,
            min_frequency=config.bpe_min_frequency,
            codebook_size=config.codebook_size,
        )

        # base:
        #   0 .. C - 1
        #
        # merges:
        #   C .. C + M - 1
        #
        # PAD:
        #   C + M
        self.pad_id = (
            config.codebook_size
            + config.bpe_num_merges
        )

        base_config = RQVAEConfig(
            **{
                field.name: getattr(
                    config,
                    field.name,
                )
                for field in fields(
                    RQVAEConfig
                )
            }
        )

        super().__init__(base_config)

    @property
    def sid_codebook_sizes(self) -> tuple[int, ...]:
        """Vocabulary capacity of every post-BPE SID position.

        BPE tokens and PAD are valid values at each original RQ position, so
        downstream models must reserve the complete BPE vocabulary, not only
        the RQ-VAE's base codebook.
        """

        return (self.bpe.vocab_size,) * self.bpe_config.num_levels

    @property
    def tracking_config(self) -> BPE_RQVAEConfig:
        """Expose the complete configuration in experiment metadata."""

        return self.bpe_config

    def train(
        self,
        df: pl.DataFrame,
    ) -> None:
        super().train(df)

        # Use the configured assignment policy (nearest/ECM/RRS) consistently
        # both while learning merges and while emitting final IDs.
        sids = self.assign(df)

        sid_columns = [
            f"sid_{i}"
            for i in range(
                self.model.num_levels
            )
        ]

        self.bpe.fit_polars(
            sids,
            sid_columns=sid_columns,
        )

    def update(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        sids = self.assign(df)

        sid_columns = [
            f"sid_{i}"
            for i in range(
                self.model.num_levels
            )
        ]

        return self.bpe.transform_polars(
            sids,
            sid_columns=sid_columns,
            pad_id=self.pad_id,
        )
