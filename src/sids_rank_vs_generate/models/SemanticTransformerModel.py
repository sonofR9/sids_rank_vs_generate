import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.config_SemanticTransformerModel import SemanticTransformerConfig
from .history_transformer_encoder import HistoryTransformerEncoder

class SemanticTransformerModel(nn.Module):
    def __init__(self, num_items, cfg):
        super().__init__()
        self.cfg = cfg or SemanticTransformerConfig()
        self.num_levels = self.cfg.num_levels
        self.codebook_size = self.cfg.codebook_size
        self.bos_id = self.codebook_size * self.num_levels
        self.encoder = HistoryTransformerEncoder(num_items, self.cfg)

        # TODO: embeddings for decoder input tokens
        self.decoder_embed = nn.Embedding(self.codebook_size*self.num_levels+1,self.cfg.d_model)
        # TODO: embeddings for semantic positions / levels
        self.positions_embed = nn.Embedding(self.num_levels,self.cfg.d_model)
        # TODO: TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.cfg.d_model,
            nhead = self.cfg.num_heads,
            dim_feedforward = self.cfg.d_ff,
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.num_layers)
        # TODO: prediction heads for each semantic level
        self.heads = nn.ModuleList([nn.Linear(self.cfg.d_model, self.codebook_size) for _ in range(self.num_levels)])

    def make_decoder_input(self, sid):
        # sid: [B, M]
        # return: [B, M] = [BOS, sid_0, sid_1, ...]
        left_part = torch.full((sid.size(0), 1), self.bos_id, device = sid.device, dtype=sid.dtype)
        right_part = sid[:,:-1]
        return torch.cat((left_part, right_part), dim=1)

    def causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def add_sid_offset(self, sid):
        T = sid.size(1)
        offsets = torch.arange(T, device=sid.device) * self.codebook_size
        return sid + offsets.unsqueeze(0)


    def forward(self, history, positions, mask, sid):
        # history: [B, L]
        # sid: [B, M]
        # TODO: encode history
        memory, _ = self.encoder(history, positions, mask)

        # TODO: build decoder input tokens
        sid_with_offset = self.add_sid_offset(sid)
        dec_tokens = self.make_decoder_input(sid_with_offset)

        # TODO: build decoder embeddings
        decoder_positions = torch.arange(
            dec_tokens.size(1),
            device=dec_tokens.device,
        ).unsqueeze(0)
        x = self.decoder_embed(dec_tokens) + self.positions_embed(decoder_positions)

        # TODO: run TransformerDecoder
        dec = self.decoder(
            tgt = x,
            memory = memory,
            tgt_mask = self.causal_mask(dec_tokens.size(1), device=sid.device),
            memory_key_padding_mask  = ~mask,
        )

        losses = []
        preds = []

        for level in range(self.num_levels):
            # TODO: predict sid[:, level] from decoder output at this level
            logits = self.heads[level](dec[:,level])
            loss = F.cross_entropy(logits, sid[:, level])

            losses.append(loss)
            preds.append(logits.argmax(dim=-1))

        return {
            "loss": torch.stack(losses).mean(),
            "preds": torch.stack(preds, dim=1)
        }