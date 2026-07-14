from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

def init_embedding(embedding, padding_idx=None):
    nn.init.trunc_normal_(
        embedding.weight,
        mean=0.0,
        std=0.02,
        a=-0.04,
        b=0.04,
    )

    if padding_idx is not None:
        with torch.no_grad():
            embedding.weight[padding_idx].zero_()


class CandidateOnlyMLP(nn.Module):
    def __init__(
        self,
        num_items,
        item_emb_dim=32,
        hidden_dim=64,
        dropout=0.1,
        pad_item_idx=0,
    ):
        super().__init__()

        self.item_emb = nn.Embedding(
            num_items,
            item_emb_dim,
            padding_idx=pad_item_idx,
        )

        init_embedding(
            self.item_emb,
            padding_idx=pad_item_idx,
        )

        # item embedding + 3 candidate признака
        input_dim = item_emb_dim + 3

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        candidate_item_idx,
        candidate_is_organic,
        candidate_track_length_z,
        candidate_has_track_length,
        **kwargs,
    ):
        candidate_vec = self.item_emb(
            candidate_item_idx
        )

        candidate_context = torch.stack(
            [
                candidate_is_organic,
                candidate_track_length_z,
                candidate_has_track_length,
            ],
            dim=-1,
        )

        x = torch.cat(
            [
                candidate_vec,
                candidate_context,
            ],
            dim=-1,
        )

        logits = self.mlp(x).squeeze(-1)

        return logits


class UserCandidateMLP(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        user_emb_dim=32,
        item_emb_dim=32,
        hidden_dim=128,
        dropout=0.1,
        pad_item_idx=0,
    ):
        super().__init__()

        self.user_emb = nn.Embedding(
            num_users,
            user_emb_dim,
        )

        self.item_emb = nn.Embedding(
            num_items,
            item_emb_dim,
            padding_idx=pad_item_idx,
        )

        init_embedding(self.user_emb)

        init_embedding(
            self.item_emb,
            padding_idx=pad_item_idx,
        )

        input_dim = (user_emb_dim + item_emb_dim + 3)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        user_idx,
        candidate_item_idx,
        candidate_is_organic,
        candidate_track_length_z,
        candidate_has_track_length,
        **kwargs,
    ):
        user_vec = self.user_emb(user_idx)

        candidate_vec = self.item_emb(
            candidate_item_idx
        )

        candidate_context = torch.stack(
            [
                candidate_is_organic,
                candidate_track_length_z,
                candidate_has_track_length,
            ],
            dim=-1,
        )

        x = torch.cat(
            [
                user_vec,
                candidate_vec,
                candidate_context,
            ],
            dim=-1,
        )

        logits = self.mlp(x).squeeze(-1)

        return logits


class RichMeanHistoryMLP(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_event_types,
        user_emb_dim=32,
        item_emb_dim=32,
        event_emb_dim=8,
        history_dim=64,
        hidden_dim=128,
        dropout=0.15,
        pad_item_idx=0,
        pad_event_idx=0,
    ):
        super().__init__()

        self.user_emb = nn.Embedding(
            num_users,
            user_emb_dim,
        )

        # Одна общая таблица и для candidate и для item в истории
        self.item_emb = nn.Embedding(
            num_items,
            item_emb_dim,
            padding_idx=pad_item_idx,
        )

        self.event_emb = nn.Embedding(
            num_event_types,
            event_emb_dim,
            padding_idx=pad_event_idx,
        )

        init_embedding(self.user_emb)

        init_embedding(
            self.item_emb,
            padding_idx=pad_item_idx,
        )

        init_embedding(
            self.event_emb,
            padding_idx=pad_event_idx,
        )

        # Числовые признаки history token:
        #
        # 1 is_organic
        # 2 played_ratio
        # 3 has_played_ratio
        # 4 track_length_z
        # 5 has_track_length
        # 6 time_delta_z
        num_history_numeric_features = 6

        history_token_input_dim = (
            item_emb_dim
            + event_emb_dim
            + num_history_numeric_features
        )

        self.history_token_encoder = nn.Sequential(
            nn.Linear(
                history_token_input_dim,
                history_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(history_dim),
            nn.Dropout(dropout),
        )

        # Candidate context:
        # is_organic, track_length_z, has_track_length
        num_candidate_features = 3

        final_input_dim = (
            user_emb_dim
            + item_emb_dim
            + num_candidate_features
            + history_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1))

    def mean_pool_history(
        self,
        history_tokens_flat,
        history_lengths,
    ):
        """
        history_tokens_flat:
            [total_history_tokens, history_dim]

        history_lengths:
            [B]

        return:
            [B, history_dim]
        """
        batch_size = history_lengths.numel()
        history_dim = history_tokens_flat.size(-1)

        history_vectors = torch.zeros(
            batch_size,
            history_dim,
            device=history_tokens_flat.device,
            dtype=history_tokens_flat.dtype,
        )

        # Для каждого history token указываем к какому примеру батча он относится
        batch_ids = torch.repeat_interleave(
            torch.arange(
                batch_size,
                device=history_lengths.device,
            ),
            history_lengths,
        )

        history_vectors.index_add_(
            dim=0,
            index=batch_ids,
            source=history_tokens_flat,
        )

        history_vectors = (
            history_vectors
            / history_lengths
            .clamp_min(1)
            .unsqueeze(-1)
        )

        return history_vectors

    def forward(
        self,
        user_idx,
        candidate_item_idx,
        candidate_is_organic,
        candidate_track_length_z,
        candidate_has_track_length,
        history_item_idxs_flat,
        history_event_type_idxs_flat,
        history_is_organic_flat,
        history_played_ratio_flat,
        history_has_played_ratio_flat,
        history_track_length_z_flat,
        history_has_track_length_flat,
        history_time_delta_z_flat,
        history_lengths,
        **kwargs,
    ):

        # Candidate и пользователь

        user_vec = self.user_emb(
            user_idx
        )  # [B, user_emb_dim]

        candidate_vec = self.item_emb(
            candidate_item_idx
        )  # [B, item_emb_dim]

        candidate_context = torch.stack(
            [
                candidate_is_organic,
                candidate_track_length_z,
                candidate_has_track_length,
            ],
            dim=-1,
        )  # [B, 3]

        # Богатые history tokens

        history_item_vec = self.item_emb(
            history_item_idxs_flat
        )  # [total_tokens, item_emb_dim]

        history_event_vec = self.event_emb(
            history_event_type_idxs_flat
        )  # [total_tokens, event_emb_dim]

        history_numeric = torch.stack(
            [
                history_is_organic_flat,
                history_played_ratio_flat,
                history_has_played_ratio_flat,
                history_track_length_z_flat,
                history_has_track_length_flat,
                history_time_delta_z_flat,
            ],
            dim=-1,
        )  # [total_tokens, 6]

        history_token_input = torch.cat(
            [
                history_item_vec,
                history_event_vec,
                history_numeric,
            ],
            dim=-1,
        )

        history_tokens = self.history_token_encoder(
            history_token_input
        )  # [total_tokens, history_dim]

        history_vec = self.mean_pool_history(
            history_tokens_flat=history_tokens,
            history_lengths=history_lengths,
        )  # [B, history_dim]

        # Финальный ранкер

        x = torch.cat(
            [
                user_vec,
                candidate_vec,
                candidate_context,
                history_vec,
            ],
            dim=-1,
        )

        logits = self.mlp(x).squeeze(-1)

        return logits


def flat_tokens_to_padded(
    tokens_flat,
    lengths,
):
    """
    tokens_flat:
        [total_history_tokens, d_model]

    lengths:
        [B]

    returns:
        padded_tokens: [B, T, d_model]
        padding_mask:  [B, T]

    В padding_mask:
        False = настоящий токен
        True  = padding
    """
    token_chunks = torch.split(
        tokens_flat,
        lengths.tolist(),
        dim=0,
    )

    padded_tokens = pad_sequence(
        token_chunks,
        batch_first=True,
        padding_value=0.0,
    )

    max_length = padded_tokens.size(1)

    positions = torch.arange(
        max_length,
        device=lengths.device,
    ).unsqueeze(0)

    padding_mask = (
        positions
        >= lengths.unsqueeze(1)
    )

    return padded_tokens, padding_mask


class CandidateTokenTransformer(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_event_types,

        user_emb_dim=32,
        item_emb_dim=32,
        event_emb_dim=8,

        d_model=64,
        n_heads=4,
        n_layers=1,
        ff_dim=256,

        hidden_dim=128,
        dropout=0.2,

        max_history_len=100,

        pad_item_idx=0,
        pad_event_idx=0,
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.max_history_len = max_history_len

        # Embeddings

        self.user_emb = nn.Embedding(
            num_users,
            user_emb_dim,
        )

        # Общая таблица для candidate и history items
        self.item_emb = nn.Embedding(
            num_items,
            item_emb_dim,
            padding_idx=pad_item_idx,
        )

        self.event_emb = nn.Embedding(
            num_event_types,
            event_emb_dim,
            padding_idx=pad_event_idx,
        )

        init_embedding(self.user_emb)

        init_embedding(
            self.item_emb,
            padding_idx=pad_item_idx,
        )

        init_embedding(
            self.event_emb,
            padding_idx=pad_event_idx,
        )

        # Candidate token

        # Candidate содержит:
        # item embedding
        # is_organic
        # track_length_z
        # has_track_length

        candidate_input_dim = (
            item_emb_dim + 3
        )

        self.candidate_token_encoder = nn.Sequential(
            nn.Linear(candidate_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # History tokens

        # Числовые признаки:
        # 1 is_organic
        # 2 played_ratio
        # 3 has_played_ratio
        # 4 track_length_z
        # 5 has_track_length
        # 6 time_delta_z

        num_history_numeric_features = 6

        history_input_dim = (item_emb_dim + event_emb_dim + num_history_numeric_features)

        self.history_token_encoder = nn.Sequential(
            nn.Linear(history_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Position и token type embeddings

        # Позиция 0  candidate token.
        # Позиции 1...L  history tokens.
        self.position_emb = nn.Embedding(max_history_len + 1, d_model)

        # 0  candidate token
        # 1  history token
        self.token_type_emb = nn.Embedding(2, d_model)

        init_embedding(self.position_emb)
        init_embedding(self.token_type_emb)

        self.input_dropout = nn.Dropout(dropout)

        # Transformer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

        # Финальный ranking head

        # Оставляем baseline признаки напрямую:
        # user embedding
        # candidate embedding
        # candidate context
        # transformer candidate state
        #
        # Благодаря этому модель может использовать
        # сильный baseline и добавить историю сверху.

        final_input_dim = (
            user_emb_dim
            + item_emb_dim
            + 3
            + d_model
        )

        self.mlp = nn.Sequential(
            nn.Linear(final_input_dim,  hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        user_idx,
        candidate_item_idx,

        candidate_is_organic,
        candidate_track_length_z,
        candidate_has_track_length,

        history_item_idxs_flat,
        history_event_type_idxs_flat,

        history_is_organic_flat,
        history_played_ratio_flat,
        history_has_played_ratio_flat,

        history_track_length_z_flat,
        history_has_track_length_flat,
        history_time_delta_z_flat,

        history_lengths,

        **kwargs,
    ):
        batch_size = user_idx.size(0)

        # 1 User и candidate

        user_vec = self.user_emb(
            user_idx
        )
        # [B, user_emb_dim]

        candidate_vec = self.item_emb(
            candidate_item_idx
        )
        # [B, item_emb_dim]

        candidate_context = torch.stack(
            [
                candidate_is_organic,
                candidate_track_length_z,
                candidate_has_track_length,
            ],
            dim=-1,
        )
        # [B, 3]

        # 2 Candidate token

        candidate_token_input = torch.cat(
            [
                candidate_vec,
                candidate_context,
            ],
            dim=-1,
        )
        # [B, item_emb_dim + 3]

        candidate_token = (
            self.candidate_token_encoder(
                candidate_token_input
            )
        )
        # [B, d_model]

        candidate_token = (
            candidate_token.unsqueeze(1)
        )
        # [B, 1, d_model]

        # 3 Rich history tokens

        history_item_vec = self.item_emb(
            history_item_idxs_flat
        )
        # [total_tokens, item_emb_dim]

        history_event_vec = self.event_emb(
            history_event_type_idxs_flat
        )
        # [total_tokens, event_emb_dim]

        history_numeric = torch.stack(
            [
                history_is_organic_flat,
                history_played_ratio_flat,
                history_has_played_ratio_flat,
                history_track_length_z_flat,
                history_has_track_length_flat,
                history_time_delta_z_flat,
            ],
            dim=-1,
        )
        # [total_tokens, 6]

        history_token_input = torch.cat(
            [
                history_item_vec,
                history_event_vec,
                history_numeric,
            ],
            dim=-1,
        )

        history_tokens_flat = (
            self.history_token_encoder(
                history_token_input
            )
        )
        # [total_tokens, d_model]

        # 4 Padding внутри encoder

        (
            history_tokens,
            history_padding_mask,
        ) = flat_tokens_to_padded(
            tokens_flat=history_tokens_flat,
            lengths=history_lengths,
        )

        # history_tokens:
        # [B, T, d_model]
        #
        # history_padding_mask:
        # [B, T]

        max_length = history_tokens.size(1)

        if max_length > self.max_history_len:
            raise ValueError(
                f"History length {max_length} "
                f"is larger than configured "
                f"max_history_len={self.max_history_len}"
            )

        # 5 Candidate + history sequence

        sequence = torch.cat(
            [
                candidate_token,
                history_tokens,
            ],
            dim=1,
        )
        # [B, 1 + T, d_model]

        # Candidate никогда не padding
        candidate_padding_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=history_padding_mask.device,
        )

        sequence_padding_mask = torch.cat(
            [
                candidate_padding_mask,
                history_padding_mask,
            ],
            dim=1,
        )
        # [B, 1 + T]

        # 6 Positions

        sequence_length = sequence.size(1)

        position_ids = torch.arange(
            sequence_length,
            device=sequence.device,
            dtype=torch.long,
        ).unsqueeze(0)

        position_vectors = self.position_emb(
            position_ids
        )
        # [1, 1 + T, d_model]

        # 7 Token types

        token_type_ids = torch.ones(
            batch_size,
            sequence_length,
            device=sequence.device,
            dtype=torch.long,
        )

        # Первый токен candidate
        token_type_ids[:, 0] = 0

        token_type_vectors = (
            self.token_type_emb(
                token_type_ids
            )
        )
        # [B, 1 + T, d_model]

        sequence = (
            sequence
            + position_vectors
            + token_type_vectors
        )

        sequence = self.input_dropout(
            sequence
        )
        # 8 Transformer

        encoded_sequence = self.transformer(
            sequence,
            src_key_padding_mask=(
                sequence_padding_mask
            ),
        )
        # [B, 1 + T, d_model]

        encoded_sequence = self.output_norm(
            encoded_sequence
        )

        # 9 Берём candidate output

        candidate_history_vec = (
            encoded_sequence[:, 0]
        )
        # [B, d_model]

        # Это candidate token обновлённый информацией из истории

        # 10 Ranking head

        final_features = torch.cat(
            [
                user_vec,
                candidate_vec,
                candidate_context,
                candidate_history_vec,
            ],
            dim=-1,
        )

        logits = self.mlp(
            final_features
        ).squeeze(-1)

        return logits
