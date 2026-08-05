import polars as pl

from dataclasses import asdict
from .base import ISidPipeline
from .plum_rq_vae import RQVAE
import torch
from torch.utils.data import DataLoader
from ..data.embedding_dataset import EmbeddingDataset
from .rq_kmeans import RQKMeans

class PlumRQVAEPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config
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
        self.model.pretrain_autoencoder(loader, tracker=self.tracker)
        # получаем выход энкодера предобуечнного
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=self.model.device)
            Z = self.model.encoder(X_tensor).cpu().numpy()
        rq = RQKMeans(codebook_sizes=tuple(self.model.codebooks_sizes))
        rq.fit(Z)
        # инитим через rqkmeans
        self.model.init_codebooks(rq.codebooks)
        self.model.fit(X, tracker=self.tracker)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)
