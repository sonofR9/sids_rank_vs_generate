import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..data.embedding_dataset import EmbeddingDataset
from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything
from ..tracking import ExperimentTracker, NullTracker


class RQVAE(nn.Module):
    def __init__(self, dim=128, hidden_dim=128, latent_dim=128, num_levels=3, codebook_size=2048, codebook_sizes=None, batch_size=256, num_epochs=50, lr=5e-4, beta=0.25, collaborative_weight=2e-2, diversity_weight=1e-4, diversity_clusters=10, num_workers=4, device=None, pretrain_epochs=3, pretrain_lr=5e-3):
        super().__init__()
        seed_everything(SID_EXPERIMENT_SEED)

        self.dim = dim
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.codebook_sizes = tuple(codebook_sizes or [codebook_size] * num_levels)
        if len(self.codebook_sizes) != num_levels:
            raise ValueError("codebook_sizes must have one entry per level")
        if any(size < 2 * diversity_clusters for size in self.codebook_sizes):
            raise ValueError("each codebook must be at least twice diversity_clusters")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta = beta
        self.collaborative_weight = collaborative_weight
        self.diversity_weight = diversity_weight
        self.diversity_clusters = diversity_clusters
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.diversity_cluster_labels = None
        self.training_metrics = {}

        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(size, self.latent_dim)) for size in self.codebook_sizes
        ])

        for codebook in self.codebooks:
            nn.init.normal_(codebook, mean=0.0, std=0.1)

        self.to(self.device)

    @staticmethod
    def converter(X, column="normalized_embed"):
        if isinstance(X, pl.DataFrame):
            X = X[column]
        if isinstance(X, pl.Series):
            X = np.stack(X.to_list())
        return np.asarray(X, dtype=np.float32)

    def pretrain_autoencoder(self, loader, tracker: ExperimentTracker | None = None):
        tracker = tracker or NullTracker()
        opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.pretrain_lr)
        self.train()

        for epoch in range(self.pretrain_epochs):
            total_loss = 0.0
            total_count = 0

            for x in loader:
                x = x.to(self.device, non_blocking=True)
                x_hat = self.decoder(self.encoder(x))
                loss = F.mse_loss(x_hat, x)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_count += bs

            mean_recon = total_loss / total_count
            print(f"pretrain epoch={epoch + 1}/{self.pretrain_epochs} | recon={mean_recon:.6f}")
            tracker.log_metrics({"pretrain.reconstruction_loss": mean_recon}, step=epoch + 1)

    @torch.no_grad()
    def init_codebooks(self, codebooks):
        if len(codebooks) != self.num_levels:
            raise ValueError("expected one initialized codebook per level")
        for level, codebook in enumerate(codebooks):
            codebook = torch.as_tensor(codebook, dtype=self.codebooks[level].dtype, device=self.device)
            if codebook.shape != self.codebooks[level].shape:
                raise ValueError(
                    f"codebook {level} has shape {tuple(codebook.shape)}, "
                    f"expected {tuple(self.codebooks[level].shape)}"
                )
            self.codebooks[level].copy_(codebook)

        self.init_diversity_clusters()

    @staticmethod
    def constrained_kmeans_labels(embeddings, num_clusters, iterations=5):
        values = embeddings.detach().float().cpu().numpy()
        count = len(values)
        if count < 2 * num_clusters:
            raise ValueError("each diversity cluster must contain at least two codebook entries")
        seed_indices = np.linspace(0, count - 1, num_clusters, dtype=np.int64)
        centers = values[seed_indices].copy()
        capacities = np.full(num_clusters, count // num_clusters, dtype=np.int64)
        capacities[:count % num_clusters] += 1
        labels = np.empty(count, dtype=np.int64)

        for _ in range(iterations):
            distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            candidates = np.argsort(distances, axis=None)
            remaining = capacities.copy()
            labels.fill(-1)

            for flat_index in candidates:
                code_idx, cluster_idx = np.unravel_index(flat_index, distances.shape)
                if labels[code_idx] == -1 and remaining[cluster_idx] > 0:
                    labels[code_idx] = cluster_idx
                    remaining[cluster_idx] -= 1
                    if not remaining.any():
                        break

            centers = np.stack([values[labels == cluster].mean(axis=0) for cluster in range(num_clusters)])

        return torch.as_tensor(labels, dtype=torch.long, device=embeddings.device)

    @torch.no_grad()
    def init_diversity_clusters(self):
        self.diversity_cluster_labels = [
            self.constrained_kmeans_labels(codebook, self.diversity_clusters) for codebook in self.codebooks
        ]

    def quantize(self, z):
        residual = z
        codes_all = []
        quantized_all = []
        residuals_all = []

        for codebook in self.codebooks:
            residuals_all.append(residual)
            distances = torch.cdist(residual, codebook, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary")
            codes = distances.argmin(dim=1)
            q = codebook[codes]

            codes_all.append(codes)
            quantized_all.append(q)
            residual = residual - q.detach()

        codes = torch.stack(codes_all, dim=1)
        z_q = torch.stack(quantized_all, dim=1).sum(dim=1)
        z_q_st = z + (z_q - z).detach()

        return z_q, z_q_st, codes, quantized_all, residuals_all

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

    def collaborative_loss(self, z_q, collaborative):
        if z_q.shape != collaborative.shape:
            raise ValueError(
                "collaborative embeddings must have the same batch shape as latent embeddings; "
                f"got {tuple(collaborative.shape)} and {tuple(z_q.shape)}"
            )
        logits = z_q @ collaborative.T
        labels = torch.arange(z_q.size(0), device=z_q.device)
        return F.cross_entropy(logits, labels)

    def diversity_loss(self, codes):
        level_losses = []

        for level, codebook in enumerate(self.codebooks):
            cluster_labels = self.diversity_cluster_labels[level]
            assigned = codes[:, level]
            positives = torch.empty_like(assigned)

            for code in assigned.unique():
                batch_mask = assigned == code
                cluster = cluster_labels[code]
                members = torch.nonzero(cluster_labels == cluster, as_tuple=False).flatten()
                members = members[members != code]
                sampled = torch.randint(len(members), (int(batch_mask.sum()),), device=self.device)
                positives[batch_mask] = members[sampled]

            logits = codebook[assigned] @ codebook.T
            logits[torch.arange(len(assigned), device=self.device), assigned] = float("-inf")
            level_losses.append(F.cross_entropy(logits, positives))

        return torch.stack(level_losses).mean()

    def compute_loss(self, x, collaborative):
        out = self(x)

        reconstruction_loss = F.mse_loss(out["x_hat"], x)
        codebook_loss = 0.0
        commitment_loss = 0.0

        for quantized, residual in zip(out["quantized_all"], out["residuals_all"]):
            codebook_loss += F.mse_loss(quantized, residual.detach())
            commitment_loss += F.mse_loss(residual, quantized.detach())

        collaborative_loss = self.collaborative_loss(out["z_q"], collaborative)
        diversity_loss = self.diversity_loss(out["codes"])

        loss = (
            reconstruction_loss
            + codebook_loss
            + self.beta * commitment_loss
            + self.collaborative_weight * collaborative_loss
            + self.diversity_weight * diversity_loss
        )

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "collaborative_loss": collaborative_loss,
            "diversity_loss": diversity_loss,
        }

    def fit(self, X, collaborative_embeddings, tracker: ExperimentTracker | None = None):
        tracker = tracker or NullTracker()
        X = self.converter(X)
        collaborative_embeddings = self.converter(collaborative_embeddings, column="collaborative_embed")

        if len(X) == 0:
            raise ValueError("cannot train RQVAE on an empty dataset")
        if len(X) != len(collaborative_embeddings):
            raise ValueError("embedding and collaborative embedding counts must match")

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(collaborative_embeddings, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=str(self.device).startswith("cuda"), num_workers=self.num_workers)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)

        names = ["loss", "reconstruction_loss", "codebook_loss", "commitment_loss", "collaborative_loss", "diversity_loss"]

        for epoch in range(self.num_epochs):
            self.train()
            totals = {name: 0.0 for name in names}
            total_count = 0

            for x, collaborative in loader:
                x = x.to(self.device, non_blocking=True)
                collaborative = collaborative.to(self.device, non_blocking=True)
                losses = self.compute_loss(x, collaborative)

                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                opt.step()

                bs = x.size(0)
                total_count += bs
                for name in names:
                    totals[name] += losses[name].item() * bs

            self.training_metrics = {name: totals[name] / total_count for name in names}
            metrics = " | ".join(f"{name}={value:.6f}" for name, value in self.training_metrics.items())
            print(f"epoch={epoch + 1}/{self.num_epochs} | {metrics}")
            tracker.log_metrics({f"train.{name}": value for name, value in self.training_metrics.items()}, step=epoch + 1)

        return self

    @torch.no_grad()
    def transform(self, df):
        X = self.converter(df)
        if len(X) == 0:
            return pl.DataFrame({"item_id": df["item_id"], **{f"sid_{level}": pl.Series([], dtype=pl.Int32) for level in range(self.num_levels)}})
        loader = DataLoader(EmbeddingDataset(X), batch_size=self.batch_size, shuffle=False, pin_memory=str(self.device).startswith("cuda"), num_workers=self.num_workers)

        self.eval()
        all_codes = []

        for x in loader:
            x = x.to(self.device, non_blocking=True)
            all_codes.append(self(x)["codes"].cpu())

        codes = torch.cat(all_codes).numpy().astype(np.int32)
        return pl.DataFrame({"item_id": df["item_id"], **{f"sid_{level}": codes[:, level] for level in range(self.num_levels)}})

    def fit_transform(self, df, collaborative_embeddings, tracker: ExperimentTracker | None = None):
        self.fit(df, collaborative_embeddings, tracker=tracker)
        return self.transform(df)
