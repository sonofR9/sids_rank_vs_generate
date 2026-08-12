import polars as pl

from dataclasses import dataclass
from ..reproducibility import seed_everything
from dataclasses import asdict
from .base import ISidPipeline
from .plum_rq_vae import RQVAE
import torch
from torch.utils.data import DataLoader
from ..data.embedding_dataset import EmbeddingDataset
from .rq_kmeans import RQKMeans

@dataclass(frozen=True)
class PlumRQVAEConfig:
    dim: int
    hidden_dim: int = 128
    latent_dim: int = 128
    num_levels: int = 3
    codebook_size: int = 2048
    codebooks_sizes: tuple[int, ...] = (512, 256)
    batch_size: int = 256
    num_epochs: int = 50
    lr: float = 5e-4
    beta: float = 0.25
    num_workers: int = 4
    device: str = "auto"
    pretrain_epochs: int = 3
    pretrain_lr: float = 5e-3

class PlumRQVAEPipeline(ISidPipeline):
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, df: pl.DataFrame) -> None:
        seed_everything(self.config.seed)
        dim = len(df["normalized_embed"][0])
        parameters = asdict(self.config)
        parameters.pop("seed", None)
        codebook_sizes = parameters.pop("codebook_sizes", None)
        if codebook_sizes is not None:
            parameters["codebooks_sizes"] = codebook_sizes
            parameters["num_levels"] = len(codebook_sizes)
        if parameters.get("device") == "auto":
            parameters["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RQVAE(dim=dim, **parameters)
    
        X = self.model.converter(df)
    
        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.model.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.model.num_workers,
        )
        # претрейним тут
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
