import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses.logq_retrieval import InBatchRetrievalLoss
from .history_transformer_encoder import HistoryTransformerEncoder

class SemanticTransformerModel(nn.Module):
    def __init__(
        self,
        num_items,
        cfg,
        initial_item_embeddings=None,
        item_sid_codes=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_levels = self.cfg.num_levels
        configured_sizes = getattr(self.cfg, "codebook_sizes", None)
        if configured_sizes is None:
            configured_sizes = (self.cfg.codebook_size,) * self.num_levels
        self.codebook_sizes = tuple(int(size) for size in configured_sizes)
        if len(self.codebook_sizes) != self.num_levels:
            raise ValueError(
                "number of levels must match the number of codebook sizes"
            )
        if any(size <= 0 for size in self.codebook_sizes):
            raise ValueError("codebook sizes must be positive")

        offsets = torch.tensor(
            [0, *self.codebook_sizes[:-1]],
            dtype=torch.long,
        ).cumsum(dim=0)
        self.register_buffer("codebook_offsets", offsets, persistent=False)
        self.codebook_size = self.codebook_sizes[0]
        self.bos_id = sum(self.codebook_sizes)
        self.encoder = HistoryTransformerEncoder(num_items, self.cfg)
        self.use_user_profile = bool(
            getattr(self.cfg, "use_user_profile", False)
        )
        self.user_profile_dim = int(
            getattr(self.cfg, "user_profile_dim", 0)
        )
        if self.use_user_profile:
            if self.user_profile_dim <= 0:
                raise ValueError(
                    "user_profile_dim must be positive when user profiles are enabled"
                )
            self.user_profile_projection = nn.Linear(
                self.user_profile_dim,
                self.cfg.d_model,
            )
            self.user_profile_norm = nn.LayerNorm(self.cfg.d_model)
        else:
            self.user_profile_projection = None
            self.user_profile_norm = None

        if initial_item_embeddings is not None:
            expected_shape = tuple(self.encoder.item_emb.weight.shape)
            if tuple(initial_item_embeddings.shape) != expected_shape:
                raise ValueError(
                    "initial item embeddings have shape "
                    f"{tuple(initial_item_embeddings.shape)}; expected {expected_shape}"
                )
            if not torch.isfinite(initial_item_embeddings).all():
                raise ValueError("initial item embeddings contain non-finite values")
            with torch.no_grad():
                self.encoder.item_emb.weight.copy_(
                    initial_item_embeddings.to(
                        device=self.encoder.item_emb.weight.device,
                        dtype=self.encoder.item_emb.weight.dtype,
                    )
                )

        # TODO: embeddings for decoder input tokens
        self.decoder_embed = nn.Embedding(self.bos_id + 1, self.cfg.d_model)
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
        self.heads = nn.ModuleList([
            nn.Linear(self.cfg.d_model, size)
            for size in self.codebook_sizes
        ])

        if item_sid_codes is not None:
            item_sid_codes = torch.as_tensor(item_sid_codes, dtype=torch.long)
            expected_shape = (num_items + 1, self.num_levels)
            if tuple(item_sid_codes.shape) != expected_shape:
                raise ValueError(
                    f"item SID codes have shape {tuple(item_sid_codes.shape)}; "
                    f"expected {expected_shape}"
                )
            for level, size in enumerate(self.codebook_sizes):
                tokens = item_sid_codes[:, level]
                if torch.any((tokens < -1) | (tokens >= size)):
                    raise ValueError(f"item SID codes at level {level} are out of range")
            self.register_buffer("item_sid_codes", item_sid_codes)
        else:
            self.item_sid_codes = None

        self.semantic_history_weight = float(
            getattr(self.cfg, "semantic_history_weight", 0.0)
        )
        self.auxiliary_item_loss_weight = float(
            getattr(self.cfg, "auxiliary_item_loss_weight", 0.0)
        )
        self.ranking_temperature = float(
            getattr(self.cfg, "ranking_temperature", 0.07)
        )
        if self.ranking_temperature <= 0:
            raise ValueError("ranking temperature must be positive")
        self.retrieval_loss = InBatchRetrievalLoss(
            temperature=self.ranking_temperature,
            logq_mode=getattr(self.cfg, "logq_mode", "none"),
            eps=float(getattr(self.cfg, "logq_eps", 1e-12)),
            debug=bool(getattr(self.cfg, "logq_debug", False)),
            gather_distributed=bool(
                getattr(self.cfg, "gather_distributed_negatives", False)
            ),
        )

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
        return sid + self.codebook_offsets[:T].unsqueeze(0)

    def _semantic_history_embeddings(self, history):
        if self.item_sid_codes is None or self.semantic_history_weight == 0:
            return None

        codes = self.item_sid_codes[history]
        valid = codes >= 0
        token_ids = codes.clamp_min(0) + self.codebook_offsets.view(1, 1, -1)
        embeddings = self.decoder_embed(token_ids)
        embeddings = embeddings * valid.unsqueeze(-1)
        embeddings = embeddings.sum(dim=2) / valid.sum(dim=2, keepdim=True).clamp_min(1)
        return embeddings * self.semantic_history_weight

    def _user_profile_context(self, history, user_features):
        if not self.use_user_profile:
            if user_features is not None:
                raise ValueError("user features were provided to a disabled model")
            return None
        if user_features is None:
            raise ValueError("user features are required by this model")
        expected_shape = (history.size(0), self.user_profile_dim)
        if tuple(user_features.shape) != expected_shape:
            raise ValueError(
                f"user features have shape {tuple(user_features.shape)}, "
                f"expected {expected_shape}"
            )
        if not user_features.is_cuda and not torch.isfinite(user_features).all():
            raise ValueError("user features must contain only finite values")
        context = self.user_profile_projection(user_features)
        return self.user_profile_norm(context)

    def encode_history(self, history, positions, mask, user_features=None):
        memory, pooled_history = self.encoder(
            history,
            positions,
            mask,
            additional_item_embeddings=self._semantic_history_embeddings(history),
        )
        user_context = self._user_profile_context(history, user_features)
        if user_context is not None:
            memory = memory + user_context.unsqueeze(1)
            pooled_history = pooled_history + user_context
        return memory, pooled_history

    def _ranking_loss(self, pooled_history, target, log_q=None):
        if target.ndim != 1 or target.size(0) != pooled_history.size(0):
            raise ValueError("target must contain one item ID per history")
        target_embeddings = self.encoder.item_emb(target)
        output = self.retrieval_loss(pooled_history, target_embeddings, log_q)
        if (
            self.retrieval_loss.logq_mode == "none"
            and output.raw_logits.size(1) == target.size(0)
        ):
            positive_mask = target.unsqueeze(1) == target.unsqueeze(0)
            positive_logits = output.raw_logits.masked_fill(
                ~positive_mask,
                float("-inf"),
            )
            output.loss = (
                torch.logsumexp(output.raw_logits, dim=1)
                - torch.logsumexp(positive_logits, dim=1)
            ).mean()
        return output

    def forward(
        self,
        history,
        positions,
        mask,
        sid,
        target=None,
        log_q=None,
        sid_log_q=None,
        user_features=None,
    ):
        # history: [B, L]
        # sid: [B, M]
        # TODO: encode history
        memory, pooled_history = self.encode_history(
            history,
            positions,
            mask,
            user_features=user_features,
        )

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

        if sid_log_q is not None and len(sid_log_q) != self.num_levels:
            raise ValueError("sid_log_q must contain one tensor per SID level")

        for level in range(self.num_levels):
            # TODO: predict sid[:, level] from decoder output at this level
            logits = self.heads[level](dec[:,level])
            loss_logits = logits
            if sid_log_q is not None:
                level_log_q = sid_log_q[level]
                if level_log_q.shape != logits.shape:
                    raise ValueError(
                        "each sid_log_q tensor must match its level logits"
                    )
                # Fixed-coefficient LogQ adjustment. There is intentionally no
                # alpha: q is part of the categorical training objective.
                loss_logits = logits.float() - level_log_q.float()
            loss = F.cross_entropy(loss_logits, sid[:, level])

            losses.append(loss)
            preds.append(loss_logits.argmax(dim=-1))

        sid_loss = torch.stack(losses).mean()
        ranking_loss = None
        ranking_diagnostics = {}
        if target is not None and self.auxiliary_item_loss_weight != 0:
            ranking_output = self._ranking_loss(pooled_history, target, log_q)
            ranking_loss = ranking_output.loss
            ranking_diagnostics = ranking_output.diagnostics
        total_loss = sid_loss
        if ranking_loss is not None:
            total_loss = total_loss + self.auxiliary_item_loss_weight * ranking_loss

        return {
            "loss": total_loss,
            "sid_loss": sid_loss,
            "ranking_loss": ranking_loss,
            "ranking_diagnostics": ranking_diagnostics,
            "preds": torch.stack(preds, dim=1),
        }
