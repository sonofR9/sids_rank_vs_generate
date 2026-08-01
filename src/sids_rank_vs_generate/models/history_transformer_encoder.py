import torch
import torch.nn as nn

class HistoryTransformerEncoder(nn.Module):
    def __init__(self, num_items, cfg):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(num_items + 1, self.cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.cfg.max_history_len, self.cfg.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.num_layers)
        self.norm = nn.LayerNorm(self.cfg.d_model)

    def forward(self, history, positions, mask):
        x = self.item_emb(history) + self.pos_emb(positions)

        x = self.encoder(
            x,
            src_key_padding_mask=~mask,
        )

        x = self.norm(x)

        pooled = (x * mask.unsqueeze(-1)).sum(1)
        pooled = pooled / mask.sum(1, keepdim=True).clamp_min(1)

        return x, pooled