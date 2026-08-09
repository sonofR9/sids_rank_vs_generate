import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..data.embedding_dataset import EmbeddingDataset
from ..tracking import ExperimentTracker, NullTracker


class RQVAE(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=128,
        latent_dim=128,
        num_levels=3,
        codebook_size=512,
        batch_size=256,
        lr=1e-3,
        beta=0.25,
        num_workers=4,
        device=None,
        pretrain_epochs=3,
        pretrain_lr=5e-3,
        max_epochs=100,
        patience=5,
        min_delta=1e-5,
        val_part=0.05,
    ):
        super().__init__()
        torch.manual_seed(SID_EXPERIMENT_SEED)
        np.random.seed(SID_EXPERIMENT_SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SID_EXPERIMENT_SEED)
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_part = val_part
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
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr

    @staticmethod
    def converter(X) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            X = X["normalized_embed"]

        if isinstance(X, pl.Series):
            X = np.stack(X.to_list())

        return np.asarray(X, dtype=np.float32)

    def compute_loss(self, x):
        out = self(x)
        recon_loss = F.mse_loss(out["x_hat"], x)
        codebook_loss = 0.0
        commit_loss = 0.0
        for q, residual in zip(out["quantized_all"], out["residuals_all"]):
            codebook_loss += F.mse_loss(q,residual.detach())
            commit_loss += F.mse_loss(residual, q.detach())
        codebook_loss /= self.num_levels
        commit_loss /= self.num_levels
        loss = (recon_loss + codebook_loss + self.beta * commit_loss)

        return loss, recon_loss, codebook_loss, commit_loss

    @torch.no_grad()
    def evaluate_metrics(self, loader):
        self.eval()
        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
        }
        total_count = 0
        for x in loader:
            x = x.to(self.device, non_blocking=True)
            loss, recon, codebook, commit = self.compute_loss(x)
            batch_size = x.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["reconstruction_loss"] += recon.item() * batch_size
            totals["codebook_loss"] += codebook.item() * batch_size
            totals["commitment_loss"] += commit.item() * batch_size
            total_count += batch_size

        return {
            name: value / max(total_count, 1)
            for name, value in totals.items()
        }

    @torch.no_grad()
    def evaluate(self, loader):
        return self.evaluate_metrics(loader)["loss"]

    def pretrain_autoencoder(
        self,
        loader,
        *,
        tracker: ExperimentTracker | None = None,
    ):
        tracker = tracker or NullTracker()
        # Оптимизируем только параметры энкодера и декодера, не заходим в кодбук
        opt = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.pretrain_lr)
        self.train()
        for epoch in range(self.pretrain_epochs):
            total_loss = 0.0
            total_count = 0
            for x in loader:
                x = x.to(self.device, non_blocking=True)
                z = self.encoder(x)
                x_hat = self.decoder(z)
                loss = F.mse_loss(x_hat, x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                bs = x.size(0)
                total_loss += loss.item() * bs
                total_count += bs
            print(
                f"pretrain epoch={epoch + 1}/{self.pretrain_epochs} | "
                f"recon_loss={total_loss/total_count}")
            tracker.log_metrics(
                {"pretrain.reconstruction_loss": total_loss / total_count},
                step=epoch + 1,
            )


    @torch.no_grad()
    def init_codebooks(self, codebooks):
        for level, codebook in enumerate(codebooks):
            codebook = torch.as_tensor(
                codebook,
                dtype=self.codebooks[level].dtype,
                device=self.device
            )
            self.codebooks[level].copy_(codebook)

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
            residual = residual - q.detach()
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
    
    def fit(
        self,
        X,
        *,
        tracker: ExperimentTracker | None = None,
    ):
        tracker = tracker or NullTracker()
        X = self.converter(X)

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X))
        val_size = int(len(X) * self.val_part)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_loader = DataLoader(
            EmbeddingDataset(X[train_indices]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        val_loader = DataLoader(
            EmbeddingDataset(X[val_indices]),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.train()

            totals = {
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "codebook_loss": 0.0,
                "commitment_loss": 0.0,
            }
            total_count = 0

            for x in train_loader:
                x = x.to(self.device, non_blocking=True)

                loss, recon, codebook, commit = (
                    self.compute_loss(x)
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = x.size(0)
                totals["loss"] += loss.item() * batch_size
                totals["reconstruction_loss"] += recon.item() * batch_size
                totals["codebook_loss"] += codebook.item() * batch_size
                totals["commitment_loss"] += commit.item() * batch_size
                total_count += batch_size

            train_metrics = {
                name: value / max(total_count, 1)
                for name, value in totals.items()
            }
            val_metrics = self.evaluate_metrics(val_loader)
            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            tracker.log_metrics(
                {
                    **{
                        f"train.{name}": value
                        for name, value in train_metrics.items()
                    },
                    **{
                        f"validation.{name}": value
                        for name, value in val_metrics.items()
                    },
                    "validation.best_loss": min(best_val_loss, val_loss),
                },
                step=epoch + 1,
            )

            print(
                f"epoch={epoch + 1}/{self.max_epochs} | "
                f"train={train_loss} | "
                f"val={val_loss} | "
                f"best={best_val_loss}"
            )

            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print("Перестало улучшаться")
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        X = self.converter(df)

        loader = DataLoader(
            EmbeddingDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=str(self.device).startswith("cuda"),
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
