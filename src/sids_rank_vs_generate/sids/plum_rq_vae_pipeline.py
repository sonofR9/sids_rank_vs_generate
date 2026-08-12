import polars as pl
import torch
from dataclasses import asdict
from torch.utils.data import DataLoader

from ..data.embedding_dataset import EmbeddingDataset
from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything
from .base import ISidPipeline
from .config import PlumRQVAEConfig
from .plum_rq_vae import RQVAE
from .rq_kmeans import RQKMeans


def _model_parameters(config: PlumRQVAEConfig) -> dict:
    parameters = asdict(config)
    parameters["codebooks_sizes"] = parameters.pop("codebook_sizes")
    if parameters.get("device") == "auto":
        parameters["device"] = None
    return parameters

class PlumRQVAEPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        seed_everything(SID_EXPERIMENT_SEED)
        dim = len(df["normalized_embed"][0])
        self.model = RQVAE(dim=dim, **_model_parameters(self.config))

        X = self.model.converter(df)

        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.model.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.model.num_workers,
        )
        self.model.pretrain_autoencoder(loader, tracker=self.tracker)
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=self.model.device)
            Z = self.model.encoder(X_tensor).cpu().numpy()
        rq = RQKMeans(codebook_sizes=tuple(self.model.codebooks_sizes))
        rq.fit(Z)
        self.model.init_codebooks(rq.codebooks)
        self.model.fit(X, tracker=self.tracker)

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise RuntimeError("Сначала вызови train()!")

        return self.model.transform(df)
