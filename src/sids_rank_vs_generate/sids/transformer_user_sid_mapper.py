from __future__ import annotations

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
    load_checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"frequency", "transformer"}:
            raise ValueError("mapper kind must be 'frequency' or 'transformer'")
        if self.load_checkpoint_path is not None and self.kind != "transformer":
            raise ValueError("only Transformer mapper can load a checkpoint")
        if self.d_model <= 0 or self.d_model % self.num_heads:
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
        )
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
    ) -> dict[str, torch.Tensor]:
        memory = self.encode(user_features, user_sids)
        losses = []
        predictions = []
        for level in range(len(self.global_sid_sizes)):
            logits = self.decode_next(memory, global_sids[:, :level], level)
            losses.append(F.cross_entropy(logits, global_sids[:, level]))
            predictions.append(logits.argmax(dim=-1))
        return {
            "loss": torch.stack(losses).mean(),
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
        self._valid_next = self._build_prefix_index(self._valid_paths)
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
            loss = self._model(features, user_sids, global_sids)["loss"]
            total += float(loss.item()) * features.size(0)
            count += features.size(0)
        return total / max(count, 1)

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
        validation_tensors = self._tensors_from_rows(
            self._rows_from_examples(validation_examples)
        )
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
        best_loss = float("inf")
        best_state = None
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
                loss = self._model(features, user_sids, global_sids)["loss"]
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
            epochs_trained = epoch
            tracker.log_metrics(
                {
                    "train.loss": train_loss,
                    "validation.loss": validation_loss,
                    "validation.best_loss": min(best_loss, validation_loss),
                },
                step=epoch,
            )
            if validation_loss < best_loss - self.config.early_stopping_min_delta:
                best_loss = validation_loss
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
                best_validation_loss=best_loss,
                epochs_trained=epochs_trained,
                train_examples=len(train_tensors[0]),
                validation_examples=len(validation_tensors[0]),
            ),
            self.checkpoint_path,
        )
        self.training_summary = {
            "train_examples": len(train_tensors[0]),
            "validation_examples": len(validation_tensors[0]),
            "epochs_trained": epochs_trained,
            "best_validation_loss": best_loss,
            "checkpoint": str(self.checkpoint_path),
        }
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
        epochs_trained: int,
        train_examples: int,
        validation_examples: int,
    ) -> dict[str, Any]:
        return {
            "model_state_dict": model_state_dict,
            "config": asdict(self.config),
            "user_sid_sizes": self.user_sid_sizes,
            "global_sid_sizes": self.global_sid_sizes,
            "best_validation_loss": best_validation_loss,
            "epochs_trained": epochs_trained,
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
        if tuple(checkpoint.get("user_sid_sizes", ())) != self.user_sid_sizes:
            raise ValueError("mapper checkpoint has incompatible personalized SID sizes")
        if tuple(checkpoint.get("global_sid_sizes", ())) != self.global_sid_sizes:
            raise ValueError(
                "mapper checkpoint has incompatible global SID sizes"
            )

        saved_config = checkpoint.get("config", {})
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
        epochs_trained = int(checkpoint.get("epochs_trained", 0))
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
                epochs_trained=epochs_trained,
                train_examples=len(train_rows),
                validation_examples=validation_examples,
            ),
            self.checkpoint_path,
        )
        self.training_summary = {
            "train_examples": len(train_rows),
            "validation_examples": validation_examples,
            "epochs_trained": epochs_trained,
            "best_validation_loss": best_validation_loss,
            "loaded_checkpoint": str(source),
            "checkpoint": str(self.checkpoint_path),
        }
        self._is_fitted = True

    @torch.inference_mode()
    def _predict_chunk(
        self,
        requests: list[tuple[object, SidPath]],
        k: int,
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
            log_probs = F.log_softmax(logits, dim=-1)
            beam_prefixes = beams.detach().cpu().tolist()
            allowed = torch.zeros(
                batch_size,
                beam_count,
                vocabulary_size,
                dtype=torch.bool,
            )
            for batch_index, sample_prefixes in enumerate(beam_prefixes):
                for beam_index, prefix_values in enumerate(sample_prefixes):
                    prefix = tuple(int(code) for code in prefix_values)
                    valid_codes = self._valid_next.get(prefix, ())
                    if valid_codes:
                        allowed[
                            batch_index,
                            beam_index,
                            list(valid_codes),
                        ] = True
            allowed = allowed.to(self.device, non_blocking=True)
            candidate_scores = (
                scores.unsqueeze(-1) + log_probs.masked_fill(~allowed, float("-inf"))
            ).flatten(1)
            next_beam_count = min(k, candidate_scores.size(1))
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
                ]
            )
        return results

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
        unique_requests: list[tuple[object, SidPath]] = []
        unique_indices: dict[tuple[object, SidPath], int] = {}
        result_indices = []
        for request in normalized:
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
                    unique_requests[
                        offset : offset + self.config.inference_batch_size
                    ],
                    k,
                )
            )
        return [list(unique_results[index]) for index in result_indices]

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
