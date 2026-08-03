import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..data.embedding_dataset import EmbeddingDataset


class PLUMRQVAE(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=128,
        latent_dim=128,
        num_levels=3,
        codebook_size=2048,
        codebook_sizes=None,
        batch_size=256,
        num_epochs=50,
        lr=5e-4,
        beta=0.25,
        num_workers=4,
        device=None,
        pretrain_epochs=3,
        pretrain_lr=5e-3,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.codebook_sizes = codebook_sizes
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
        if self.codebook_sizes is None:
            self.codebook_sizes = [self.codebook_size // 2**i for i in range(num_levels)]
        if len(self.codebook_sizes) != self.num_levels:
            raise ValueError(
                "Количество уровней должно совпадать с количеством codebook_sizes"
            )
        # TODO: residual quantization codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(size, latent_dim))
            for size in self.codebook_sizes
        ])

        for codebook in self.codebooks:
            nn.init.normal_(codebook, mean=0.0, std=0.1)

        self.to(self.device)
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.metric_logger = None

    @staticmethod
    def converter(X) -> np.ndarray:
        if isinstance(X, pl.DataFrame):
            X = X["normalized_embed"]

        if isinstance(X, pl.Series):
            X = np.stack(X.to_list())

        return np.asarray(X, dtype=np.float32)

    def pretrain_autoencoder(self, loader):
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
            epoch_metrics = {
                "plum/pretrain_reconstruction_loss": total_loss / total_count,
            }
            if self.metric_logger is not None:
                self.metric_logger(epoch_metrics, step=epoch + 1)
            print(
                f"pretrain epoch={epoch + 1}/{self.pretrain_epochs} | "
                f"recon={total_loss / total_count:.6f}"
            )

    def sample_mask(self, batch_size, dtype, device):
        # закрепляю вероятность ласт кодбука, а остальную размазываю равномерно
        min_last_probability = 0.25
        probs = torch.full(
            (self.num_levels - 1,),
            (1.0 - min_last_probability) / (self.num_levels - 1),
            device=device)
        probs = torch.cat([probs, torch.tensor([min_last_probability], device=device)])
        # семплирую согласно вероятностям
        sample = torch.multinomial(probs, num_samples=batch_size, replacement=True)+1
        # маска для каждого айтем
        mask = torch.stack([torch.arange(self.num_levels, device=device) < item for item in sample]).to(dtype) # [batch_size, num_levels]
        return mask
        
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
            residual = residual - q.detach()

        codes_all = torch.stack(codes_all, dim=1) # [B, L]
        
        quantized_stack = torch.stack(quantized_all, dim=1) #[B, L, D]
        residuals_stack = torch.stack(residuals_all, dim=1) # [B, L, D]
        if self.training:
            mask = self.sample_mask(
                batch_size=z.size(0),
                dtype=z.dtype,
                device=z.device)
        else:
            mask = torch.ones(
                z.size(0),
                self.num_levels,
                dtype=z.dtype,
                device=z.device)

        z_q = (quantized_stack * mask.unsqueeze(-1)).sum(dim=1)
        z_q_st = z + (z_q - z).detach()
        return z_q, z_q_st, codes_all, quantized_stack, residuals_stack, mask

    def forward(self, x):
        z = self.encoder(x)
        z_q, z_q_st, codes, quantized_stack, residuals_stack, mask = self.quantize(z)
        x_hat = self.decoder(z_q_st)

        return {
            "x_hat": x_hat,
            "z": z,
            "z_q": z_q,
            "codes": codes,
            "quantized_stack": quantized_stack,
            "residuals_stack": residuals_stack,
            "mask": mask
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_codebook = 0.0
            total_commit = 0.0
            total_count = 0
    
            for x in loader:
                # на recon_loss не влияет маска, тут всё также
                x = x.to(self.device, non_blocking=True)
                out = self(x)
    
                x_hat = out["x_hat"]
                quantized_stack  = out["quantized_stack"]
                residuals_stack = out["residuals_stack"]
                mask = out['mask']
    
                recon_loss = F.mse_loss(x, x_hat)
                ''' 
                а вот лоссы кодбука и commit_loss надо переписать
                посчитаю фулл лосс без усреднения, потом постепенно усредню 
                и уберу маскированную часть
                '''
                full_codebook_loss = F.mse_loss(
                    quantized_stack, 
                    residuals_stack.detach(),
                    reduction="none"
                ).mean(dim=-1) # усреднения по длине эмбеда
                full_commit_loss = F.mse_loss(
                    quantized_stack.detach(), 
                    residuals_stack,
                    reduction="none"
                ).mean(dim=-1)# усреднения по длине эмбеда
                active_mask_part = mask.sum() # тут мегахайп, мы усредним одновременно и по работающим кодбукам и по батчу
                codebook_loss = (full_codebook_loss*mask).sum()/active_mask_part
                commit_loss = (full_commit_loss*mask).sum()/active_mask_part
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
    
            epoch_metrics = {
                "plum/loss": total_loss / total_count,
                "plum/reconstruction_loss": total_recon / total_count,
                "plum/codebook_loss": total_codebook / total_count,
                "plum/commitment_loss": total_commit / total_count,
            }
            if self.metric_logger is not None:
                self.metric_logger(epoch_metrics, step=epoch + 1)
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
