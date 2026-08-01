from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn

from ..data.ranking import build_ranking_loaders, prepare_ranking_data
from ..models.dcn_v2.dcn_v2 import DCNV2
from ..models.transact import CandidateTokenTransformer, init_embedding
from ..reproducibility import seed_everything
from .dcn_metrics import compute_pairwise_accuracy
from .stages import EvaluationContext, IEvaluationStage


RANKING_MODELS = ("dcnv2", "transact")


@dataclass(frozen=True)
class RankingConfig:
    models: tuple[str, ...] = RANKING_MODELS
    max_history_len: int = 50
    min_history_len: int = 2
    negatives_per_positive: int = 1
    validation_fraction: float = 0.1
    max_train_examples: int | None = None
    max_eval_examples: int | None = 100_000
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 32
    hidden_dim: int = 128
    dcn_cross_layers: int = 3
    dcn_deep_units: tuple[int, ...] = (256, 128, 64)
    dcn_num_experts: int = 4
    dcn_low_rank: int = 32
    transact_d_model: int = 64
    transact_heads: int = 4
    transact_layers: int = 1
    transact_ff_dim: int = 256
    dropout: float = 0.2
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        unknown = set(self.models) - set(RANKING_MODELS)
        if unknown:
            raise ValueError(f"unknown ranking models: {sorted(unknown)}")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("ranking models must be non-empty and unique")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.transact_d_model % self.transact_heads:
            raise ValueError("transact_d_model must be divisible by transact_heads")


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class SemanticItemEmbedding(nn.Module):
    """Drop-in item embedding backed only by per-level semantic codes."""

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
        padded = torch.zeros(
            sid_codes.size(0) + 1, sid_codes.size(1), dtype=torch.long
        )
        padded[1:] = sid_codes + torch.tensor(offsets, dtype=torch.long)
        self.register_buffer("item_to_tokens", padded)
        self.embedding = nn.Embedding(offset, embedding_dim, padding_idx=0)
        init_embedding(self.embedding, padding_idx=0)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        tokens = self.item_to_tokens[item_ids]
        return self.embedding(tokens).mean(dim=-2)


class DCNRankingAdapter(nn.Module):
    def __init__(
        self,
        num_users: int,
        sid_cardinalities: tuple[int, ...],
        config: RankingConfig,
    ) -> None:
        super().__init__()
        offsets = []
        offset = num_users
        for cardinality in sid_cardinalities:
            offsets.append(offset)
            offset += cardinality
        self.register_buffer("sid_offsets", torch.tensor(offsets, dtype=torch.long))
        self.feature_names = ("uid",) + tuple(
            f"sid_{level}" for level in range(len(sid_cardinalities))
        )
        input_size = config.embedding_dim * len(self.feature_names)
        self.model = DCNV2(
            embedding_size=config.embedding_dim,
            cross_layers=config.dcn_cross_layers,
            deep_units=list(config.dcn_deep_units),
            input_size=input_size,
            dense_train_df=None,
            n_bins=2,
            train_df_slice=0,
            cardinality=offset,
            num_experts=config.dcn_num_experts,
            low_rank=min(config.dcn_low_rank, input_size),
            deep_network="mlp",
            output_size=1,
            full_input_size_provided=True,
            sparse_feature_names=self.feature_names,
            multivalent_feature_names=(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sparse = {"uid": batch["user_idx"]}
        encoded_sids = batch["candidate_sids"] + self.sid_offsets
        for level in range(encoded_sids.size(1)):
            sparse[f"sid_{level}"] = encoded_sids[:, level]
        return self.model({"sparse_features": sparse}).squeeze(-1)


class TransActRankingAdapter(nn.Module):
    def __init__(
        self,
        num_users: int,
        sid_codes: torch.Tensor,
        sid_cardinalities: tuple[int, ...],
        config: RankingConfig,
    ) -> None:
        super().__init__()
        model = CandidateTokenTransformer(
            num_users=num_users,
            num_items=sid_codes.size(0) + 1,
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
        model.item_emb = SemanticItemEmbedding(
            sid_codes, sid_cardinalities, config.embedding_dim
        )
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            key: value
            for key, value in batch.items()
            if key not in {"target", "candidate_sids", "candidate_timestamp"}
        }
        return self.model(**inputs)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    users: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "logloss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "pairwise_accuracy": float(
            compute_pairwise_accuracy(users, timestamps, targets, probabilities)
        ),
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits_all = []
    targets_all = []
    users_all = []
    timestamps_all = []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits_all.append(model(batch).float().cpu())
        targets_all.append(batch["target"].cpu())
        users_all.append(batch["user_idx"].cpu())
        timestamps_all.append(batch["candidate_timestamp"].cpu())
    logits = torch.cat(logits_all)
    return _metrics(
        torch.cat(targets_all).numpy(),
        torch.sigmoid(logits).numpy(),
        torch.cat(users_all).numpy(),
        torch.cat(timestamps_all).numpy(),
    )


def _train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    test_loader,
    config: RankingConfig,
    artifact_path: Path,
) -> dict[str, Any]:
    device = _device(config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_auc = float("-inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * batch["target"].size(0)
            count += batch["target"].size(0)
        validation = _evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "validation": validation,
            }
        )
        if validation["roc_auc"] > best_auc:
            best_auc = validation["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("ranking training produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, artifact_path)
    return {
        "device": str(device),
        "history": history,
        "test": _evaluate(model, test_loader, device),
        "checkpoint": artifact_path,
    }


class RankingStage(IEvaluationStage):
    def __init__(self, config: RankingConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "ranking"

    def evaluate(self, context: EvaluationContext) -> dict[str, Any]:
        data = prepare_ranking_data(
            context.prepared,
            context.items_with_sids,
            min_history_len=self.config.min_history_len,
            negatives_per_positive=self.config.negatives_per_positive,
            validation_fraction=self.config.validation_fraction,
            max_train_examples=self.config.max_train_examples,
            max_eval_examples=self.config.max_eval_examples,
            seed=context.seed,
        )
        loaders = build_ranking_loaders(
            data,
            max_history_len=self.config.max_history_len,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        results = {}
        for model_name in self.config.models:
            seed_everything(context.seed)
            if model_name == "dcnv2":
                model = DCNRankingAdapter(
                    data.num_users, data.sid_cardinalities, self.config
                )
            else:
                model = TransActRankingAdapter(
                    data.num_users,
                    data.sid_codes,
                    data.sid_cardinalities,
                    self.config,
                )
            results[model_name] = _train_model(
                model,
                *loaders,
                self.config,
                context.artifact_dir / f"{model_name}.pt",
            )
        return {
            "config": asdict(self.config),
            "examples": {
                "train": len(data.train_examples),
                "validation": len(data.validation_examples),
                "test": len(data.test_examples),
            },
            "models": results,
        }
