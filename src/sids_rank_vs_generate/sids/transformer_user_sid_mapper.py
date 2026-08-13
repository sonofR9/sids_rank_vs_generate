from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..reproducibility import SID_EXPERIMENT_SEED, seed_everything
from ..tracking import ExperimentTracker, NullTracker
from .collision_resolution import get_sid_cols
from .user_aware import user_sid_columns
from .user_sid_mapper import IUserSidMapper, SidPath


MAPPER_CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class UserSidMapperConfig:
    """Настройки частотного и Transformer мапперов"""

    kind: Literal["frequency", "transformer"] = "frequency"
    d_model: int = 128
    d_ff: int = 256
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 512
    inference_batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 20
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    grad_clip: float = 1.0
    catalog_masked_teacher_forced_loss: bool = False
    level_loss_weights: tuple[float, ...] | None = None
    collision_level_loss_weight: float | None = None
    validation_selection_metric: Literal[
        "teacher_forced_loss",
        "autoregressive_catalog_top_k_accuracy",
    ] = "teacher_forced_loss"
    validation_selection_top_k: int = 3
    decode_beam_size: int | None = None
    load_checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"frequency", "transformer"}:
            raise ValueError("mapper kind must be 'frequency' or 'transformer'")
        if self.load_checkpoint_path is not None and self.kind != "transformer":
            raise ValueError("only Transformer mapper can load a checkpoint")
        if self.num_heads <= 0 or self.d_model <= 0 or self.d_model % self.num_heads:
            raise ValueError(
                "mapper d_model must be positive and divisible by num_heads"
            )
        positive = (
            self.d_ff,
            self.num_heads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.batch_size,
            self.inference_batch_size,
            self.max_epochs,
            self.early_stopping_patience,
            self.validation_selection_top_k,
        )
        if self.decode_beam_size is not None:
            positive = (*positive, self.decode_beam_size)
        if any(value <= 0 for value in positive):
            raise ValueError("mapper layer counts, sizes, and epochs must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("mapper dropout must be in [0, 1)")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("mapper lr must be positive and weight_decay non-negative")
        if self.grad_clip <= 0 or self.early_stopping_min_delta < 0:
            raise ValueError(
                "mapper grad_clip must be positive and min_delta non-negative"
            )
        if self.level_loss_weights is not None:
            weights = tuple(float(value) for value in self.level_loss_weights)
            if (
                not weights
                or any(not math.isfinite(value) or value < 0 for value in weights)
                or not sum(weights)
            ):
                raise ValueError(
                    "mapper level loss weights must be non-negative with positive sum"
                )
            object.__setattr__(self, "level_loss_weights", weights)
        if self.collision_level_loss_weight is not None:
            collision_weight = float(self.collision_level_loss_weight)
            if not math.isfinite(collision_weight) or collision_weight < 0:
                raise ValueError("mapper collision loss weight must be non-negative")
            if self.level_loss_weights is not None:
                raise ValueError(
                    "use either per-level weights or collision loss weight, not both"
                )
            object.__setattr__(
                self,
                "collision_level_loss_weight",
                collision_weight,
            )
        if self.validation_selection_metric not in {
            "teacher_forced_loss",
            "autoregressive_catalog_top_k_accuracy",
        }:
            raise ValueError("unsupported mapper validation selection metric")
        if (
            self.decode_beam_size is not None
            and self.decode_beam_size < self.validation_selection_top_k
        ):
            raise ValueError(
                "mapper decode beam size must cover validation selection top-k"
            )


class _ConditionalSidTransformer(nn.Module):
    def __init__(
        self,
        *,
        profile_dim: int,
        user_sid_sizes: Sequence[int],
        global_sid_sizes: Sequence[int],
        config: UserSidMapperConfig,
    ) -> None:
        super().__init__()
        self.user_sid_sizes = tuple(user_sid_sizes)
        self.global_sid_sizes = tuple(global_sid_sizes)
        self.profile_projection = nn.Linear(profile_dim, config.d_model)
        self.user_sid_embeddings = nn.ModuleList(
            nn.Embedding(size, config.d_model) for size in self.user_sid_sizes
        )
        self.encoder_positions = nn.Embedding(
            len(self.user_sid_sizes) + 1, config.d_model
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )
        self.bos = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.bos, std=0.02)
        self.global_sid_embeddings = nn.ModuleList(
            nn.Embedding(size, config.d_model) for size in self.global_sid_sizes
        )
        self.decoder_positions = nn.Embedding(
            len(self.global_sid_sizes), config.d_model
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )
        self.heads = nn.ModuleList(
            nn.Linear(config.d_model, size) for size in self.global_sid_sizes
        )

    def encode(
        self,
        user_features: torch.Tensor,
        user_sids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = [self.profile_projection(user_features).unsqueeze(1)]
        tokens.extend(
            embedding(user_sids[:, level]).unsqueeze(1)
            for level, embedding in enumerate(self.user_sid_embeddings)
        )
        sequence = torch.cat(tokens, dim=1)
        positions = torch.arange(sequence.size(1), device=sequence.device)
        return self.encoder(sequence + self.encoder_positions(positions).unsqueeze(0))

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )

    def _decoder_input(self, prefix: torch.Tensor) -> torch.Tensor:
        batch_size, prefix_length = prefix.shape
        tokens = [self.bos.expand(batch_size, -1).unsqueeze(1)]
        tokens.extend(
            self.global_sid_embeddings[level](prefix[:, level]).unsqueeze(1)
            for level in range(prefix_length)
        )
        sequence = torch.cat(tokens, dim=1)
        positions = torch.arange(sequence.size(1), device=sequence.device)
        return sequence + self.decoder_positions(positions).unsqueeze(0)

    def decode_next(
        self,
        memory: torch.Tensor,
        prefix: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        decoder_input = self._decoder_input(prefix)
        decoded = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=self._causal_mask(decoder_input.size(1), decoder_input.device),
        )
        return self.heads[level](decoded[:, -1])

    def forward(
        self,
        user_features: torch.Tensor,
        user_sids: torch.Tensor,
        global_sids: torch.Tensor,
        *,
        allowed_next_masks: Sequence[torch.Tensor] | None = None,
        level_loss_weights: Sequence[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        level_count = len(self.global_sid_sizes)
        if allowed_next_masks is not None and len(allowed_next_masks) != level_count:
            raise ValueError("catalog masks must contain one tensor per SID level")
        if level_loss_weights is not None and len(level_loss_weights) != level_count:
            raise ValueError("loss weights must contain one value per SID level")
        memory = self.encode(user_features, user_sids)
        losses = []
        predictions = []
        for level in range(level_count):
            logits = self.decode_next(memory, global_sids[:, :level], level)
            loss_logits = logits
            if allowed_next_masks is not None:
                allowed = allowed_next_masks[level]
                if allowed.shape != logits.shape or allowed.dtype != torch.bool:
                    raise ValueError("catalog mask has an unexpected shape or dtype")
                # Полные target paths заранее проверены на CPU
                loss_logits = logits.masked_fill(~allowed, float("-inf"))
            losses.append(F.cross_entropy(loss_logits, global_sids[:, level]))
            predictions.append(loss_logits.argmax(dim=-1))
        stacked_losses = torch.stack(losses)
        if level_loss_weights is None:
            loss = stacked_losses.mean()
        else:
            weights = torch.as_tensor(
                level_loss_weights,
                dtype=stacked_losses.dtype,
                device=stacked_losses.device,
            )
            loss = (stacked_losses * weights).sum() / weights.sum()
        return {
            "loss": loss,
            "level_losses": stacked_losses,
            "preds": torch.stack(predictions, dim=1),
        }


class TransformerUserSidMapper(IUserSidMapper):
    """Обучаемый маппер с декодированием по каталогу"""

    def __init__(
        self,
        config: UserSidMapperConfig,
        *,
        user_sid_sizes: Sequence[int],
        global_sid_sizes: Sequence[int],
        valid_global_sids: Iterable[Sequence[int]],
        user_features: pl.DataFrame,
        device: str,
        checkpoint_path: Path,
        content_sid_levels: int | None = None,
    ) -> None:
        self.config = config
        self.user_sid_sizes = tuple(int(size) for size in user_sid_sizes)
        self.global_sid_sizes = tuple(int(size) for size in global_sid_sizes)
        if not self.user_sid_sizes or not self.global_sid_sizes:
            raise ValueError("mapper SID codebook sizes must not be empty")
        if any(size <= 0 for size in (*self.user_sid_sizes, *self.global_sid_sizes)):
            raise ValueError("mapper SID codebook sizes must be positive")
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self._features, self._unknown_feature = self._load_user_features(user_features)
        self._valid_paths = self._load_valid_paths(valid_global_sids)
        self._valid_path_set = set(self._valid_paths)
        self._valid_next = self._build_prefix_index(self._valid_paths)
        self.content_sid_levels = content_sid_levels
        if content_sid_levels is not None and not isinstance(content_sid_levels, int):
            raise ValueError("content SID levels must be an integer")
        self._level_loss_weights = self._resolve_level_loss_weights(content_sid_levels)
        if self._level_loss_weights is not None and len(
            self._level_loss_weights
        ) != len(self.global_sid_sizes):
            raise ValueError("mapper level loss weights must match global SID levels")
        self.catalog_fingerprint = self._catalog_fingerprint(self._valid_paths)
        (
            self._catalog_prefix_keys,
            self._catalog_allowed_masks,
        ) = self._build_catalog_tensor_index()
        self._model = _ConditionalSidTransformer(
            profile_dim=len(self._unknown_feature),
            user_sid_sizes=self.user_sid_sizes,
            global_sid_sizes=self.global_sid_sizes,
            config=config,
        ).to(self.device)
        self.user_sid_columns: list[str] = []
        self.global_sid_columns: list[str] = []
        self.training_summary: dict[str, float | int | str] = {}
        self._seen_keys: set[tuple[object, SidPath]] = set()
        self._is_fitted = False

    def _resolve_level_loss_weights(
        self,
        content_sid_levels: int | None,
    ) -> tuple[float, ...] | None:
        if self.config.level_loss_weights is not None:
            return self.config.level_loss_weights
        collision_weight = self.config.collision_level_loss_weight
        if collision_weight is None:
            return None
        if content_sid_levels is None:
            raise ValueError(
                "content SID levels are required for collision loss weighting"
            )
        if not 0 < content_sid_levels < len(self.global_sid_sizes):
            raise ValueError(
                "content SID levels must identify a non-empty collision suffix"
            )
        return (1.0,) * content_sid_levels + (float(collision_weight),) * (
            len(self.global_sid_sizes) - content_sid_levels
        )

    def _catalog_fingerprint(self, paths: Sequence[SidPath]) -> str:
        digest = hashlib.sha256(b"user-sid-mapper-catalog-v2")
        for size in self.global_sid_sizes:
            digest.update(struct.pack(">Q", size))
        digest.update(struct.pack(">Q", len(paths)))
        for path in paths:
            for code in path:
                digest.update(struct.pack(">Q", code))
        return digest.hexdigest()

    def _requires_v2_checkpoint(self) -> bool:
        return any(
            (
                self.config.catalog_masked_teacher_forced_loss,
                self.config.level_loss_weights is not None,
                self.config.collision_level_loss_weight is not None,
                self.config.validation_selection_metric != "teacher_forced_loss",
                self.config.decode_beam_size is not None,
            )
        )

    @staticmethod
    def _load_user_features(
        frame: pl.DataFrame,
    ) -> tuple[dict[object, np.ndarray], np.ndarray]:
        required = {"uid", "user_feature"}
        if not required.issubset(frame.columns):
            raise ValueError("user features must contain uid and user_feature")
        if frame.is_empty() or frame["uid"].n_unique() != frame.height:
            raise ValueError("user features must contain one row per uid")
        features: dict[object, np.ndarray] = {}
        for uid, values in frame.select("uid", "user_feature").iter_rows():
            vector = np.asarray(values, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("user_feature must be a finite non-empty vector")
            features[uid] = vector
        dimensions = {len(vector) for vector in features.values()}
        if len(dimensions) != 1:
            raise ValueError("all user_feature vectors must have the same size")
        unknown = np.stack(list(features.values())).mean(axis=0).astype(np.float32)
        return features, unknown

    def _load_valid_paths(
        self,
        paths: Iterable[Sequence[int]],
    ) -> tuple[SidPath, ...]:
        result = sorted({tuple(int(code) for code in path) for path in paths})
        if not result:
            raise ValueError("valid global SID catalog must not be empty")
        for path in result:
            if len(path) != len(self.global_sid_sizes):
                raise ValueError("valid global SID has an unexpected number of levels")
            if any(
                code < 0 or code >= self.global_sid_sizes[level]
                for level, code in enumerate(path)
            ):
                raise ValueError("valid global SID contains an out-of-range code")
        return tuple(result)

    @staticmethod
    def _build_prefix_index(
        paths: Sequence[SidPath],
    ) -> dict[SidPath, tuple[int, ...]]:
        index: dict[SidPath, set[int]] = {}
        for path in paths:
            for level, code in enumerate(path):
                index.setdefault(path[:level], set()).add(code)
        return {prefix: tuple(sorted(codes)) for prefix, codes in index.items()}

    def _encode_prefix(self, prefix: Sequence[int]) -> int:
        result = 0
        for level, code in enumerate(prefix):
            result = result * self.global_sid_sizes[level] + int(code)
        return result

    def _build_catalog_tensor_index(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        prefix_keys = []
        allowed_masks = []
        for level, vocabulary_size in enumerate(self.global_sid_sizes):
            entries = sorted(
                (
                    self._encode_prefix(prefix),
                    codes,
                )
                for prefix, codes in self._valid_next.items()
                if len(prefix) == level
            )
            if not entries:
                raise ValueError("catalog has no valid prefix for a SID level")
            keys = torch.tensor(
                [key for key, _codes in entries],
                dtype=torch.long,
                device=self.device,
            )
            masks = torch.zeros(
                len(entries),
                vocabulary_size,
                dtype=torch.bool,
            )
            for row_index, (_key, codes) in enumerate(entries):
                masks[row_index, list(codes)] = True
            prefix_keys.append(keys)
            allowed_masks.append(masks.to(self.device, non_blocking=True))
        return tuple(prefix_keys), tuple(allowed_masks)

    def _catalog_allowed_next(
        self,
        prefixes: torch.Tensor,
        level: int,
        *,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prefixes.ndim != 2 or prefixes.size(1) != level:
            raise ValueError("catalog prefixes have an unexpected shape")
        encoded = torch.zeros(
            prefixes.size(0),
            dtype=torch.long,
            device=prefixes.device,
        )
        for prefix_level in range(level):
            encoded = (
                encoded * self.global_sid_sizes[prefix_level]
                + prefixes[:, prefix_level]
            )
        keys = self._catalog_prefix_keys[level]
        positions = torch.searchsorted(keys, encoded)
        in_range = positions < keys.numel()
        safe_positions = positions.clamp_max(keys.numel() - 1)
        matched = in_range & (keys.index_select(0, safe_positions) == encoded)
        has_active_mask = active is not None
        if active is None:
            active = torch.ones_like(matched)
        elif active.shape != matched.shape or active.dtype != torch.bool:
            raise ValueError("catalog active-prefix mask has an unexpected shape")
        allowed = self._catalog_allowed_masks[level].index_select(0, safe_positions)
        # На CPU эти проверки помогают тестам, на CUDA paths уже гарантированы
        if prefixes.device.type == "cpu":
            if not bool((matched | ~active).all()):
                raise ValueError("global SID prefix is outside the catalog")
            if not bool((allowed.any(dim=1) | ~active).all()):
                raise ValueError("catalog prefix has no valid next token")
        if has_active_mask:
            allowed = allowed.clone()
            allowed[~active] = False
            allowed[~active, 0] = True
        return allowed

    def _feature(self, uid: object) -> np.ndarray:
        return self._features.get(uid, self._unknown_feature)

    def _tensors_from_rows(
        self,
        rows: list[tuple[object, Sequence[int], Sequence[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.tensor(
            np.stack([self._feature(uid) for uid, _, _ in rows]),
            dtype=torch.float32,
        )
        user_sids = torch.tensor(
            [list(user_sid) for _, user_sid, _ in rows], dtype=torch.long
        )
        global_sids = torch.tensor(
            [list(global_sid) for _, _, global_sid in rows], dtype=torch.long
        )
        if user_sids.shape[1] != len(self.user_sid_sizes):
            raise ValueError("personalized SID has an unexpected number of levels")
        if global_sids.shape[1] != len(self.global_sid_sizes):
            raise ValueError("global SID has an unexpected number of levels")
        for level, size in enumerate(self.user_sid_sizes):
            if (user_sids[:, level] < 0).any() or (user_sids[:, level] >= size).any():
                raise ValueError("personalized SID contains an out-of-range code")
        for level, size in enumerate(self.global_sid_sizes):
            if (global_sids[:, level] < 0).any() or (
                global_sids[:, level] >= size
            ).any():
                raise ValueError("global SID contains an out-of-range code")
        if self.config.catalog_masked_teacher_forced_loss:
            paths = {tuple(int(code) for code in path) for path in global_sids.tolist()}
            if not paths.issubset(self._valid_path_set):
                raise ValueError("mapper target global SID is outside the catalog")
        return features, user_sids, global_sids

    def _rows_from_frame(
        self,
        frame: pl.DataFrame,
    ) -> list[tuple[object, Sequence[int], Sequence[int]]]:
        self.user_sid_columns = user_sid_columns(frame)
        self.global_sid_columns = get_sid_cols(frame)
        if not self.user_sid_columns or not self.global_sid_columns:
            raise ValueError("mapper training requires usid_* and sid_* columns")
        columns = ["uid", *self.user_sid_columns, *self.global_sid_columns]
        rows: list[tuple[object, Sequence[int], Sequence[int]]] = []
        split = 1 + len(self.user_sid_columns)
        for row in frame.select(*columns).iter_rows():
            rows.append((row[0], row[1:split], row[split:]))
        return rows

    @staticmethod
    def _rows_from_examples(
        examples: list[dict],
    ) -> list[tuple[object, Sequence[int], Sequence[int]]]:
        return [
            (example["uid"], example["sid"], example["global_sid"])
            for example in examples
        ]

    def _teacher_forced_catalog_masks(
        self,
        global_sids: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            self._catalog_allowed_next(global_sids[:, :level], level)
            for level in range(len(self.global_sid_sizes))
        ]

    def _model_output(
        self,
        features: torch.Tensor,
        user_sids: torch.Tensor,
        global_sids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        masks = None
        if self.config.catalog_masked_teacher_forced_loss:
            masks = self._teacher_forced_catalog_masks(global_sids)
        return self._model(
            features,
            user_sids,
            global_sids,
            allowed_next_masks=masks,
            level_loss_weights=self._level_loss_weights,
        )

    @torch.inference_mode()
    def _loss(self, tensors: tuple[torch.Tensor, ...]) -> float:
        self._model.eval()
        total = 0.0
        count = 0
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=self.config.inference_batch_size,
            shuffle=False,
        )
        for features, user_sids, global_sids in loader:
            features = features.to(self.device)
            user_sids = user_sids.to(self.device)
            global_sids = global_sids.to(self.device)
            loss = self._model_output(features, user_sids, global_sids)["loss"]
            total += float(loss.item()) * features.size(0)
            count += features.size(0)
        return total / max(count, 1)

    @torch.inference_mode()
    def _validation_catalog_top_k_accuracy(
        self,
        rows: list[tuple[object, Sequence[int], Sequence[int]]],
        *,
        k: int,
    ) -> float:
        self._model.eval()
        requests = [
            (uid, tuple(int(code) for code in user_sid))
            for uid, user_sid, _global_sid in rows
        ]
        predictions = self._predict_normalized_requests(requests, k)
        hits = sum(
            tuple(int(code) for code in global_sid) in ranked
            for ranked, (_uid, _user_sid, global_sid) in zip(
                predictions,
                rows,
                strict=True,
            )
        )
        return hits / max(len(rows), 1)

    def fit(
        self,
        frame: pl.DataFrame,
        validation_examples: list[dict],
        *,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        tracker = tracker or NullTracker()
        if frame.is_empty():
            raise ValueError("mapper training data must not be empty")
        if not validation_examples:
            raise ValueError("Transformer mapper requires validation examples")
        train_rows = self._rows_from_frame(frame)
        train_tensors = self._tensors_from_rows(train_rows)
        validation_rows = self._rows_from_examples(validation_examples)
        validation_tensors = self._tensors_from_rows(validation_rows)
        seed_everything(SID_EXPERIMENT_SEED)
        generator = torch.Generator().manual_seed(SID_EXPERIMENT_SEED)
        loader = DataLoader(
            TensorDataset(*train_tensors),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        select_by_top_k = (
            self.config.validation_selection_metric
            == "autoregressive_catalog_top_k_accuracy"
        )
        best_selection_value = float("-inf") if select_by_top_k else float("inf")
        best_validation_loss = float("inf")
        best_validation_top_k_accuracy: float | None = None
        lowest_validation_loss = float("inf")
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_trained = 0
        for epoch in range(1, self.config.max_epochs + 1):
            self._model.train()
            loss_sum = 0.0
            example_count = 0
            for features, user_sids, global_sids in loader:
                features = features.to(self.device)
                user_sids = user_sids.to(self.device)
                global_sids = global_sids.to(self.device)
                loss = self._model_output(features, user_sids, global_sids)["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.config.grad_clip
                )
                optimizer.step()
                loss_sum += float(loss.item()) * features.size(0)
                example_count += features.size(0)
            train_loss = loss_sum / max(example_count, 1)
            validation_loss = self._loss(validation_tensors)
            lowest_validation_loss = min(lowest_validation_loss, validation_loss)
            validation_top_k_accuracy = None
            if select_by_top_k:
                validation_top_k_accuracy = self._validation_catalog_top_k_accuracy(
                    validation_rows,
                    k=self.config.validation_selection_top_k,
                )
                selection_value = validation_top_k_accuracy
                improved = (
                    selection_value
                    > best_selection_value + self.config.early_stopping_min_delta
                )
                best_logged_value = max(best_selection_value, selection_value)
            else:
                selection_value = validation_loss
                improved = (
                    selection_value
                    < best_selection_value - self.config.early_stopping_min_delta
                )
                best_logged_value = min(best_selection_value, selection_value)
            epochs_trained = epoch
            epoch_metrics = {
                "train.loss": train_loss,
                "validation.loss": validation_loss,
                "validation.best_loss": (
                    lowest_validation_loss if select_by_top_k else best_logged_value
                ),
            }
            if validation_top_k_accuracy is not None:
                metric_name = (
                    "validation.autoregressive_catalog_top_"
                    f"{self.config.validation_selection_top_k}_accuracy"
                )
                epoch_metrics[metric_name] = validation_top_k_accuracy
                epoch_metrics[
                    f"validation.best_{metric_name.removeprefix('validation.')}"
                ] = best_logged_value
            tracker.log_metrics(epoch_metrics, step=epoch)
            if improved:
                best_selection_value = selection_value
                best_validation_loss = validation_loss
                best_validation_top_k_accuracy = validation_top_k_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self._model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("Transformer mapper training produced no checkpoint")
        self._model.load_state_dict(best_state)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._checkpoint_payload(
                best_state,
                best_validation_loss=best_validation_loss,
                best_validation_selection_value=best_selection_value,
                best_validation_top_k_accuracy=(best_validation_top_k_accuracy),
                epochs_trained=epochs_trained,
                best_epoch=best_epoch,
                train_examples=len(train_tensors[0]),
                validation_examples=len(validation_tensors[0]),
            ),
            self.checkpoint_path,
        )
        self.training_summary = {
            "train_examples": len(train_tensors[0]),
            "validation_examples": len(validation_tensors[0]),
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "validation_selection_metric": (self.config.validation_selection_metric),
            "best_validation_selection_value": best_selection_value,
            "checkpoint_version": MAPPER_CHECKPOINT_VERSION,
            "catalog_fingerprint": self.catalog_fingerprint,
            "checkpoint": str(self.checkpoint_path),
        }
        if best_validation_top_k_accuracy is not None:
            self.training_summary[
                "best_validation_autoregressive_catalog_top_k_accuracy"
            ] = best_validation_top_k_accuracy
            self.training_summary["validation_selection_top_k"] = (
                self.config.validation_selection_top_k
            )
        self._seen_keys = {
            (uid, tuple(int(code) for code in user_sid))
            for uid, user_sid, _ in train_rows
        }
        self._is_fitted = True

    def _checkpoint_payload(
        self,
        model_state_dict: dict[str, torch.Tensor],
        *,
        best_validation_loss: float,
        best_validation_selection_value: float,
        best_validation_top_k_accuracy: float | None,
        epochs_trained: int,
        best_epoch: int,
        train_examples: int,
        validation_examples: int,
    ) -> dict[str, Any]:
        return {
            "checkpoint_version": MAPPER_CHECKPOINT_VERSION,
            "provenance": {
                "implementation": "TransformerUserSidMapper",
                "catalog_fingerprint": self.catalog_fingerprint,
                "content_sid_levels": self.content_sid_levels,
                "profile_dim": len(self._unknown_feature),
                "effective_level_loss_weights": self._level_loss_weights,
            },
            "catalog_fingerprint": self.catalog_fingerprint,
            "model_state_dict": model_state_dict,
            "config": asdict(self.config),
            "user_sid_sizes": self.user_sid_sizes,
            "global_sid_sizes": self.global_sid_sizes,
            "best_validation_loss": best_validation_loss,
            "validation_selection_metric": (self.config.validation_selection_metric),
            "validation_selection_top_k": (self.config.validation_selection_top_k),
            "best_validation_selection_value": (best_validation_selection_value),
            "best_validation_top_k_accuracy": (best_validation_top_k_accuracy),
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "train_examples": train_examples,
            "validation_examples": validation_examples,
        }

    def load(self, frame: pl.DataFrame, checkpoint_path: Path) -> None:
        if frame.is_empty():
            raise ValueError("mapper training data must not be empty")
        source = Path(checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(f"mapper checkpoint does not exist: {source}")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("mapper checkpoint must contain a mapping")
        checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
        if not 1 <= checkpoint_version <= MAPPER_CHECKPOINT_VERSION:
            raise ValueError("unsupported mapper checkpoint version")
        if self._requires_v2_checkpoint() and checkpoint_version < 2:
            raise ValueError("mapper v2 configuration requires a version 2 checkpoint")
        if checkpoint_version >= 2:
            saved_fingerprint = checkpoint.get("catalog_fingerprint")
            if saved_fingerprint != self.catalog_fingerprint:
                raise ValueError("mapper checkpoint catalog fingerprint mismatch")
        if tuple(checkpoint.get("user_sid_sizes", ())) != self.user_sid_sizes:
            raise ValueError(
                "mapper checkpoint has incompatible personalized SID sizes"
            )
        if tuple(checkpoint.get("global_sid_sizes", ())) != self.global_sid_sizes:
            raise ValueError("mapper checkpoint has incompatible global SID sizes")

        saved_config = checkpoint.get("config", {})
        if not isinstance(saved_config, dict):
            raise ValueError("mapper checkpoint config must be a mapping")
        architecture_fields = (
            "d_model",
            "d_ff",
            "num_heads",
            "num_encoder_layers",
            "num_decoder_layers",
        )
        for name in architecture_fields:
            if saved_config.get(name) != getattr(self.config, name):
                raise ValueError(
                    f"mapper checkpoint has incompatible config field: {name}"
                )
        if checkpoint_version >= 2:
            behavior_fields = (
                "catalog_masked_teacher_forced_loss",
                "level_loss_weights",
                "collision_level_loss_weight",
                "validation_selection_metric",
                "validation_selection_top_k",
                "decode_beam_size",
            )
            for name in behavior_fields:
                saved_value = saved_config.get(name)
                current_value = getattr(self.config, name)
                if name == "level_loss_weights" and saved_value is not None:
                    saved_value = tuple(float(value) for value in saved_value)
                if saved_value != current_value:
                    raise ValueError(
                        "mapper checkpoint has incompatible v2 config field: " f"{name}"
                    )
            provenance = checkpoint.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError("mapper v2 checkpoint lacks provenance")
            if provenance.get("content_sid_levels") != self.content_sid_levels:
                raise ValueError(
                    "mapper checkpoint has incompatible content SID levels"
                )
            if provenance.get("profile_dim") != len(self._unknown_feature):
                raise ValueError("mapper checkpoint has incompatible profile dimension")
            saved_weights = provenance.get("effective_level_loss_weights")
            if saved_weights is not None:
                saved_weights = tuple(float(value) for value in saved_weights)
            if saved_weights != self._level_loss_weights:
                raise ValueError(
                    "mapper checkpoint has incompatible effective loss weights"
                )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("mapper checkpoint does not contain model_state_dict")
        self._model.load_state_dict(state_dict)
        self._model.eval()

        train_rows = self._rows_from_frame(frame)
        self._seen_keys = {
            (uid, tuple(int(code) for code in user_sid))
            for uid, user_sid, _ in train_rows
        }
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_validation_selection_value = float(
            checkpoint.get(
                "best_validation_selection_value",
                best_validation_loss,
            )
        )
        best_validation_top_k_accuracy_value = checkpoint.get(
            "best_validation_top_k_accuracy"
        )
        best_validation_top_k_accuracy = (
            float(best_validation_top_k_accuracy_value)
            if best_validation_top_k_accuracy_value is not None
            else None
        )
        epochs_trained = int(checkpoint.get("epochs_trained", 0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        validation_examples = int(checkpoint.get("validation_examples", 0))
        current_state = {
            name: value.detach().cpu().clone()
            for name, value in self._model.state_dict().items()
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._checkpoint_payload(
                current_state,
                best_validation_loss=best_validation_loss,
                best_validation_selection_value=(best_validation_selection_value),
                best_validation_top_k_accuracy=(best_validation_top_k_accuracy),
                epochs_trained=epochs_trained,
                best_epoch=best_epoch,
                train_examples=len(train_rows),
                validation_examples=validation_examples,
            ),
            self.checkpoint_path,
        )
        self.training_summary = {
            "train_examples": len(train_rows),
            "validation_examples": validation_examples,
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "validation_selection_metric": checkpoint.get(
                "validation_selection_metric",
                "teacher_forced_loss",
            ),
            "best_validation_selection_value": (best_validation_selection_value),
            "checkpoint_version": checkpoint_version,
            "catalog_fingerprint": self.catalog_fingerprint,
            "loaded_checkpoint": str(source),
            "checkpoint": str(self.checkpoint_path),
        }
        if best_validation_top_k_accuracy is not None:
            self.training_summary[
                "best_validation_autoregressive_catalog_top_k_accuracy"
            ] = best_validation_top_k_accuracy
            self.training_summary["validation_selection_top_k"] = int(
                checkpoint.get(
                    "validation_selection_top_k",
                    self.config.validation_selection_top_k,
                )
            )
        self._is_fitted = True

    @torch.inference_mode()
    def _predict_chunk(
        self,
        requests: list[tuple[object, SidPath]],
        *,
        return_k: int,
        beam_size: int,
    ) -> list[list[SidPath]]:
        features = torch.tensor(
            np.stack([self._feature(uid) for uid, _ in requests]),
            dtype=torch.float32,
            device=self.device,
        )
        user_sids = torch.tensor(
            [list(path) for _, path in requests],
            dtype=torch.long,
            device=self.device,
        )
        memory = self._model.encode(features, user_sids)
        batch_size = len(requests)
        beams = torch.empty(batch_size, 1, 0, dtype=torch.long, device=self.device)
        scores = torch.zeros(batch_size, 1, device=self.device)
        for level, vocabulary_size in enumerate(self.global_sid_sizes):
            beam_count = beams.size(1)
            flat_prefix = beams.reshape(batch_size * beam_count, level)
            logits = self._model.decode_next(
                memory.repeat_interleave(beam_count, dim=0),
                flat_prefix,
                level,
            ).reshape(batch_size, beam_count, vocabulary_size)
            allowed = self._catalog_allowed_next(
                flat_prefix,
                level,
                active=torch.isfinite(scores).reshape(-1),
            ).reshape(
                batch_size,
                beam_count,
                vocabulary_size,
            )
            masked_logits = logits.masked_fill(~allowed, float("-inf"))
            log_probs = F.log_softmax(masked_logits, dim=-1)
            candidate_scores = (scores.unsqueeze(-1) + log_probs).flatten(1)
            next_beam_count = min(beam_size, candidate_scores.size(1))
            scores, best_indices = torch.topk(
                candidate_scores, k=next_beam_count, dim=1
            )
            previous_beam = best_indices // vocabulary_size
            next_token = best_indices % vocabulary_size
            selected = beams.gather(
                1,
                previous_beam.unsqueeze(-1).expand(-1, -1, level),
            )
            beams = torch.cat([selected, next_token.unsqueeze(-1)], dim=-1)
        beam_values = beams.detach().cpu().tolist()
        finite_scores = torch.isfinite(scores).detach().cpu().tolist()
        results = []
        for sample_beams, sample_finite in zip(
            beam_values,
            finite_scores,
            strict=True,
        ):
            results.append(
                [
                    tuple(int(code) for code in path)
                    for path, is_finite in zip(
                        sample_beams,
                        sample_finite,
                        strict=True,
                    )
                    if is_finite
                ][:return_k]
            )
        return results

    def _predict_normalized_requests(
        self,
        requests: list[tuple[object, SidPath]],
        return_k: int,
    ) -> list[list[SidPath]]:
        beam_size = max(return_k, self.config.decode_beam_size or return_k)
        unique_requests: list[tuple[object, SidPath]] = []
        unique_indices: dict[tuple[object, SidPath], int] = {}
        result_indices = []
        for request in requests:
            index = unique_indices.get(request)
            if index is None:
                index = len(unique_requests)
                unique_indices[request] = index
                unique_requests.append(request)
            result_indices.append(index)

        unique_results = []
        for offset in range(0, len(unique_requests), self.config.inference_batch_size):
            unique_results.extend(
                self._predict_chunk(
                    unique_requests[offset : offset + self.config.inference_batch_size],
                    return_k=return_k,
                    beam_size=beam_size,
                )
            )
        return [list(unique_results[index]) for index in result_indices]

    def predict_topk_batch(
        self,
        requests: list[tuple[object, Iterable[int]]],
        k: int,
    ) -> list[list[SidPath]]:
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before predict_topk_batch()")
        if k <= 0:
            raise ValueError("k must be positive")
        if not requests:
            return []
        normalized = [
            (uid, tuple(int(code) for code in user_sid)) for uid, user_sid in requests
        ]
        for _, path in normalized:
            if len(path) != len(self.user_sid_sizes):
                raise ValueError("personalized SID has an unexpected number of levels")
            if any(
                code < 0 or code >= self.user_sid_sizes[level]
                for level, code in enumerate(path)
            ):
                raise ValueError("personalized SID contains an out-of-range code")
        return self._predict_normalized_requests(normalized, k)

    def predict_topk(
        self,
        uid: object,
        user_sid: Iterable[int],
        k: int,
    ) -> list[SidPath]:
        return self.predict_topk_batch([(uid, user_sid)], k)[0]

    def is_seen(self, uid: object, user_sid: Iterable[int]) -> bool:
        path = tuple(int(code) for code in user_sid)
        return (uid, path) in self._seen_keys
