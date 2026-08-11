import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.logq_retrieval import InBatchRetrievalLoss
from .history_transformer_encoder import HistoryTransformerEncoder


class BPESemanticTransformerModel(nn.Module):
    def __init__(
        self,
        num_items,
        cfg,
        initial_item_embeddings=None,
        item_sid_codes=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.num_levels = cfg.num_levels

        # sid_vocab_size включает PAD, но НЕ BOS.
        self.sid_vocab_size = int(cfg.sid_vocab_size)
        self.pad_id = int(cfg.sid_pad_id)

        # BOS нужен только как input decoder-а.
        self.bos_id = self.sid_vocab_size

        if not 0 <= self.pad_id < self.sid_vocab_size:
            raise ValueError("sid_pad_id must be inside SID vocabulary")

        self.encoder = HistoryTransformerEncoder(
            num_items,
            self.cfg,
        )

        # ---------------------------------------------------------
        # Initial item embeddings
        # ---------------------------------------------------------

        if initial_item_embeddings is not None:
            expected_shape = tuple(
                self.encoder.item_emb.weight.shape
            )

            if tuple(initial_item_embeddings.shape) != expected_shape:
                raise ValueError(
                    "initial item embeddings have shape "
                    f"{tuple(initial_item_embeddings.shape)}; "
                    f"expected {expected_shape}"
                )

            if not torch.isfinite(initial_item_embeddings).all():
                raise ValueError(
                    "initial item embeddings contain non-finite values"
                )

            with torch.no_grad():
                self.encoder.item_emb.weight.copy_(
                    initial_item_embeddings.to(
                        device=self.encoder.item_emb.weight.device,
                        dtype=self.encoder.item_emb.weight.dtype,
                    )
                )

        # ---------------------------------------------------------
        # BPE decoder
        #
        # 0 ... sid_vocab_size-1 = BPE tokens + PAD
        # sid_vocab_size         = BOS
        # ---------------------------------------------------------

        self.decoder_embed = nn.Embedding(
            self.sid_vocab_size + 1,
            self.cfg.d_model,
        )

        self.positions_embed = nn.Embedding(
            self.num_levels,
            self.cfg.d_model,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.d_ff,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.cfg.num_layers,
        )

        # Один общий head для всех позиций.
        self.head = nn.Linear(
            self.cfg.d_model,
            self.sid_vocab_size,
        )

        # ---------------------------------------------------------
        # item -> BPE SID
        #
        # Всегда:
        # [num_items + 1, num_levels]
        #
        # например:
        # [501, 8, PAD]
        # [712, PAD, PAD]
        # ---------------------------------------------------------

        if item_sid_codes is not None:
            item_sid_codes = torch.as_tensor(
                item_sid_codes,
                dtype=torch.long,
            )

            expected_shape = (
                num_items + 1,
                self.num_levels,
            )

            if tuple(item_sid_codes.shape) != expected_shape:
                raise ValueError(
                    f"item SID codes have shape "
                    f"{tuple(item_sid_codes.shape)}; "
                    f"expected {expected_shape}"
                )

            if torch.any(
                (item_sid_codes < 0)
                | (item_sid_codes >= self.sid_vocab_size)
            ):
                raise ValueError(
                    "BPE SID token is outside vocabulary"
                )

            self.register_buffer(
                "item_sid_codes",
                item_sid_codes,
            )

        else:
            self.item_sid_codes = None

        # ---------------------------------------------------------
        # Auxiliary retrieval
        # ---------------------------------------------------------

        self.semantic_history_weight = float(
            getattr(
                self.cfg,
                "semantic_history_weight",
                0.0,
            )
        )

        self.auxiliary_item_loss_weight = float(
            getattr(
                self.cfg,
                "auxiliary_item_loss_weight",
                0.0,
            )
        )

        self.ranking_temperature = float(
            getattr(
                self.cfg,
                "ranking_temperature",
                0.07,
            )
        )

        if self.ranking_temperature <= 0:
            raise ValueError(
                "ranking temperature must be positive"
            )

        self.retrieval_loss = InBatchRetrievalLoss(
            temperature=self.ranking_temperature,
            logq_mode=getattr(
                self.cfg,
                "logq_mode",
                "none",
            ),
            eps=float(
                getattr(
                    self.cfg,
                    "logq_eps",
                    1e-12,
                )
            ),
            debug=bool(
                getattr(
                    self.cfg,
                    "logq_debug",
                    False,
                )
            ),
            gather_distributed=bool(
                getattr(
                    self.cfg,
                    "gather_distributed_negatives",
                    False,
                )
            ),
        )

    def make_decoder_input(self, sid):
        # target:
        # [A, B, PAD]
        #
        # input:
        # [BOS, A, B]

        bos = torch.full(
            (sid.size(0), 1),
            self.bos_id,
            device=sid.device,
            dtype=sid.dtype,
        )

        return torch.cat(
            [bos, sid[:, :-1]],
            dim=1,
        )

    def causal_mask(self, T, device):
        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=device,
            ),
            diagonal=1,
        )

        return mask.masked_fill(
            mask == 1,
            float("-inf"),
        )

    def _semantic_history_embeddings(self, history):
        if (
            self.item_sid_codes is None
            or self.semantic_history_weight == 0
        ):
            return None

        # [B, H, num_levels]
        codes = self.item_sid_codes[history]

        # PAD не учитываем при усреднении.
        valid = codes != self.pad_id

        embeddings = self.decoder_embed(codes)

        embeddings = (
            embeddings
            * valid.unsqueeze(-1)
        )

        embeddings = (
            embeddings.sum(dim=2)
            / valid.sum(
                dim=2,
                keepdim=True,
            ).clamp_min(1)
        )

        return (
            embeddings
            * self.semantic_history_weight
        )

    def encode_history(
        self,
        history,
        positions,
        mask,
    ):
        return self.encoder(
            history,
            positions,
            mask,
            additional_item_embeddings=(
                self._semantic_history_embeddings(
                    history
                )
            ),
        )

    def _ranking_loss(
        self,
        pooled_history,
        target,
        log_q=None,
    ):
        if (
            target.ndim != 1
            or target.size(0) != pooled_history.size(0)
        ):
            raise ValueError(
                "target must contain one item ID per history"
            )

        target_embeddings = (
            self.encoder.item_emb(target)
        )

        return self.retrieval_loss(
            pooled_history,
            target_embeddings,
            log_q,
        )

    def forward(
        self,
        history,
        positions,
        mask,
        sid,
        target=None,
        log_q=None,
    ):
        # sid всегда [B, num_levels]
        #
        # например:
        # [501, 8, PAD]

        if sid.ndim != 2 or sid.size(1) != self.num_levels:
            raise ValueError(
                f"expected SID shape [B, {self.num_levels}], "
                f"got {tuple(sid.shape)}"
            )

        memory, pooled_history = self.encode_history(
            history,
            positions,
            mask,
        )

        dec_tokens = self.make_decoder_input(sid)

        decoder_positions = torch.arange(
            self.num_levels,
            device=sid.device,
        ).unsqueeze(0)

        x = (
            self.decoder_embed(dec_tokens)
            + self.positions_embed(
                decoder_positions
            )
        )

        dec = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.causal_mask(
                self.num_levels,
                sid.device,
            ),
            memory_key_padding_mask=~mask.bool(),
        )

        # [B, num_levels, vocab]
        logits = self.head(dec)

        # ВАЖНО:
        # PAD НЕ игнорируем.
        #
        # Модель должна научиться:
        # [AB, C, PAD]
        # и сама предсказывать PAD на последней позиции.
        sid_loss = F.cross_entropy(
            logits.reshape(
                -1,
                self.sid_vocab_size,
            ),
            sid.reshape(-1),
        )

        ranking_loss = None
        ranking_diagnostics = {}

        if (
            target is not None
            and self.auxiliary_item_loss_weight != 0
        ):
            ranking_output = self._ranking_loss(
                pooled_history,
                target,
                log_q,
            )

            ranking_loss = ranking_output.loss
            ranking_diagnostics = (
                ranking_output.diagnostics
            )

        total_loss = sid_loss

        if ranking_loss is not None:
            total_loss = (
                total_loss
                + self.auxiliary_item_loss_weight
                * ranking_loss
            )

        return {
            "loss": total_loss,
            "sid_loss": sid_loss,
            "ranking_loss": ranking_loss,
            "ranking_diagnostics": ranking_diagnostics,
            "preds": logits.argmax(dim=-1),
        }
    