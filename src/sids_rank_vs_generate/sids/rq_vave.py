import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..data.embedding_dataset import EmbeddingDataset
from .rq_kmeans import RQKMeans


class RQVAE(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=128,
        latent_dim=128,
        num_levels=3,
        codebook_size=512,
        batch_size=256,
        num_epochs=50,
        lr=1e-3,
        beta=0.25,
        num_workers=4,
        device=None,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta = beta
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: encoder: x -> z
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # TODO: decoder: z_q -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        # TODO: residual quantization codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(codebook_size, latent_dim))
            for _ in range(num_levels)
        ])

        for codebook in self.codebooks:
            nn.init.normal_(codebook, mean=0.0, std=0.1)

        self.to(self.device)

    @staticmethod
    def converter(X) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            X = X["normalized_embed"]

        if isinstance(X, pl.Series):
            X = np.stack(X.to_list())

        return np.asarray(X, dtype=np.float32)

    @torch.no_grad()
    def init_codebooks_rqkmeans(self, X: pl.Series | pl.DataFrame | np.ndarray):
        X = self.converter(X)
        X = torch.tensor(X, device = self.device)
        rq = RQKMeans(
            num_levels=self.num_levels,
            codebook_size=self.codebook_size)
        Z = self.encoder(X).cpu().numpy()
        rq.fit(Z)
        for level in range(rq.num_levels):
            self.codebooks[level].copy_(torch.from_numpy(rq.codebooks[level]).to(self.device))

    def quantize(self, z):
        residual = z
        z_q = torch.zeros_like(z)
        codes_all = []
        quantized_all = []
        residuals_all = []

        for codebook in self.codebooks:
            residuals_all.append(residual)

            distances = torch.cdist(
                residual,
                codebook,
                p=2,
                compute_mode="use_mm_for_euclid_dist_if_necessary")

            codes = distances.argmin(dim=1)
            q = codebook[codes]

            codes_all.append(codes)
            quantized_all.append(q)

            z_q += q
            residual = residual - q
        codes_all = torch.stack(codes_all, dim=1)
        z_q_st = z + (z_q - z).detach()
        return z_q, z_q_st, codes_all, quantized_all, residuals_all

    def forward(self, x):
        z = self.encoder(x)
        z_q, z_q_st, codes, quantized_all, residuals_all = self.quantize(z)
        x_hat = self.decoder(z_q_st)

        return {
            "x_hat": x_hat,
            "z": z,
            "z_q": z_q,
            "codes": codes,
            "quantized_all": quantized_all,
            "residuals_all": residuals_all,
        }
    
    def fit(self, X: pl.Series | pl.DataFrame | np.ndarray):
        X = self.converter(X)
    
        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
    
        self.init_codebooks_rqkmeans(X)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
    
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_codebook = 0.0
            total_commit = 0.0
            total_count = 0
    
            for x in loader:
                x = x.to(self.device, non_blocking=True)
    
                out = self(x)
    
                x_hat = out["x_hat"]
                quantized_all = out["quantized_all"]
                residuals_all = out["residuals_all"]
    
                recon_loss = F.mse_loss(x, x_hat)
                codebook_loss = 0.0
                commit_loss = 0.0
    
                for q, residual in zip(quantized_all, residuals_all):
                    codebook_loss += F.mse_loss(q, residual.detach())
                    commit_loss += F.mse_loss(residual, q.detach())
    
                codebook_loss /= self.num_levels
                commit_loss /= self.num_levels
                loss = recon_loss + codebook_loss + self.beta * commit_loss
    
                opt.zero_grad()
                loss.backward()
                opt.step()
    
                bs = x.size(0)
                total_loss += loss.item() * bs
                total_recon += recon_loss.item() * bs
                total_codebook += codebook_loss.item() * bs
                total_commit += commit_loss.item() * bs
                total_count += bs
    
            print(
                f"epoch={epoch + 1}/{self.num_epochs} | "
                f"loss={total_loss / total_count:.6f} | "
                f"recon={total_recon / total_count:.6f} | "
                f"codebook={total_codebook / total_count:.6f} | "
                f"commit={total_commit / total_count:.6f}"
            )
    
        return self

    @torch.no_grad()
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        X = self.converter(df)

        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device == "cuda",
            num_workers=self.num_workers,
        )

        self.eval()
        all_codes = []

        for x in loader:
            x = x.to(self.device, non_blocking=True)
            all_codes.append(self(x)["codes"].cpu())

        codes = torch.cat(all_codes).numpy().astype(np.int32)

        return pl.DataFrame({
            "item_id": df["item_id"],
            **{
                f"sid_{level}": codes[:, level]
                for level in range(self.num_levels)
            },
        })
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)