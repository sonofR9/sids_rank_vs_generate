import numpy as np
import polars as pl

from ..data import PreparedInteractions
from ..experiments.sasrec import SASRecConfig, run_sasrec


def build_collaborative_embeddings(
    prepared: PreparedInteractions,
    *,
    embedding_dim: int = 128,
    column: str = "collaborative_embed",
) -> pl.DataFrame:

    config = SASRecConfig(
        embedding_dim=embedding_dim,
    )

    model, _ = run_sasrec(
        prepared,
        config,
    )

    embeddings = (
        model.item_embeddings.weight[:prepared.catalog_size]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return pl.DataFrame(
        {
            "item_id": np.arange(prepared.catalog_size),
            column: embeddings.tolist(),
        }
    )

