from __future__ import annotations

import torch
from torch import nn

from ..data.ranking import (
    HASH_CARDINALITY,
    MULTIVALENT_COLUMNS,
    NUM_HASHES,
    RankingData,
)
from .dcn_v2.dcn_v2 import DCNV2
from .transact import CandidateTokenTransformer, init_embedding


class SidEmbedding(nn.Module):
    """Item embedding obtained from the item's semantic-ID tokens."""

    def __init__(
        self,
        sid_codes: torch.Tensor,
        cardinalities: tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        offsets = []
        offset = 1
        for cardinality in cardinalities:
            offsets.append(offset)
            offset += cardinality
        padded_codes = torch.zeros(
            sid_codes.size(0) + 1,
            sid_codes.size(1),
            dtype=torch.long,
        )
        padded_codes[1:] = sid_codes + torch.tensor(offsets, dtype=torch.long)
        self.register_buffer("item_to_tokens", padded_codes)
        self.embedding = nn.Embedding(offset, embedding_dim, padding_idx=0)
        init_embedding(self.embedding, padding_idx=0)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.item_to_tokens[item_ids]).mean(dim=-2)


def _dcn_input_size(
    num_sid_levels: int,
    embedding_dim: int,
    additional_size: int = 0,
) -> int:
    sparse_embeddings = (1 + num_sid_levels) * embedding_dim
    multivalent_embeddings = len(MULTIVALENT_COLUMNS) * NUM_HASHES * embedding_dim
    return sparse_embeddings + multivalent_embeddings + additional_size


def _base_dcn(
    data: RankingData,
    config,
    *,
    additional_size: int = 0,
) -> DCNV2:
    model = DCNV2(
        embedding_size=config.embedding_dim,
        cross_layers=config.dcn_cross_layers,
        deep_units=list(config.dcn_deep_units),
        input_size=_dcn_input_size(
            len(data.sid_columns),
            config.embedding_dim,
            additional_size,
        ),
        dense_train_df=data.dense_train,
        n_bins=config.dense_bins,
        train_df_slice=config.dense_train_rows,
        cardinality=HASH_CARDINALITY,
        num_experts=config.dcn_num_experts,
        low_rank=config.dcn_low_rank,
        deep_network="mlp",
        output_size=1,
        full_input_size_provided=False,
        sparse_feature_names=("uid", *data.sid_columns),
        multivalent_feature_names=MULTIVALENT_COLUMNS,
    )
    for layer in model.cross_network.layers:
        for expert in range(layer.num_experts):
            nn.init.xavier_uniform_(layer.U[expert])
            nn.init.xavier_uniform_(layer.V[expert])
        nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(layer.gate.weight)
        nn.init.zeros_(layer.gate.bias)
    return model


class DcnV2Ranker(nn.Module):
    def __init__(self, data: RankingData, config) -> None:
        super().__init__()
        self.model = _base_dcn(data, config)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch).squeeze(-1)


class TransActDcnV2Ranker(nn.Module):
    """The common DCNv2 feature set augmented with TransAct history state."""

    def __init__(self, data: RankingData, config) -> None:
        super().__init__()
        self.history_encoder = CandidateTokenTransformer(
            num_users=data.num_users,
            num_items=data.sid_codes.size(0) + 1,
            num_event_types=3,
            user_emb_dim=config.embedding_dim,
            item_emb_dim=config.embedding_dim,
            event_emb_dim=8,
            d_model=config.transact_d_model,
            n_heads=config.transact_heads,
            n_layers=config.transact_layers,
            ff_dim=config.transact_ff_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            max_history_len=config.max_history_len,
        )
        self.history_encoder.item_emb = SidEmbedding(
            data.sid_codes,
            data.sid_cardinalities,
            config.embedding_dim,
        )
        self.history_encoder.mlp = nn.Identity()
        self.ranker = _base_dcn(
            data,
            config,
            additional_size=config.transact_d_model,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_features = self.history_encoder(
            user_idx=batch["user_idx"],
            candidate_item_idx=batch["candidate_item_idx"],
            candidate_is_organic=batch["candidate_is_organic"],
            candidate_track_length_z=batch["candidate_track_length_z"],
            candidate_has_track_length=batch["candidate_has_track_length"],
            history_item_idxs_flat=batch["history_item_idxs_flat"],
            history_event_type_idxs_flat=batch["history_event_type_idxs_flat"],
            history_is_organic_flat=batch["history_is_organic_flat"],
            history_played_ratio_flat=batch["history_played_ratio_flat"],
            history_has_played_ratio_flat=batch["history_has_played_ratio_flat"],
            history_track_length_z_flat=batch["history_track_length_z_flat"],
            history_has_track_length_flat=batch["history_has_track_length_flat"],
            history_time_delta_z_flat=batch["history_time_delta_z_flat"],
            history_lengths=batch["history_lengths"],
            return_features=True,
        )
        dcn_inputs = {
            "dense_features": batch["dense_features"],
            "sparse_features": batch["sparse_features"],
            "multivalent_features": batch["multivalent_features"],
            "additional_features": history_features["history_embedding"],
        }
        return self.ranker(dcn_inputs).squeeze(-1)
