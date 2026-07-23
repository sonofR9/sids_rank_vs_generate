import polars as pl

from dataclasses import asdict
from .base import ISidPipeline
from .rq_vae import RQVAE
from experiments.Config_rqvae_rqkmeans import RQVAEConfig
import torch
from torch.utils.data import DataLoader
from ..data.embedding_dataset import EmbeddingDataset
from .rq_kmeans import RQKMeans

class RQVAEPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config or RQVAEConfig()
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        dim = len(df["normalized_embed"][0])
        self.model = RQVAE(dim=dim, **asdict(self.config))
    
        X = self.model.converter(df)
    
        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.model.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.model.num_workers,
        )
        # претрейнем тут
        self.model.pretrain_autoencoder(loader)
        # получаем выход энкодера предобуечнного
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=self.model.device)
            Z = self.model.encoder(X_tensor).cpu().numpy()
        rq = RQKMeans(num_levels=self.config.num_levels, codebook_size=self.config.codebook_size)
        rq.fit(Z)
        # инитим через rqkmeans
        self.model.init_codebooks(rq.codebooks)
        self.model.fit(X)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)