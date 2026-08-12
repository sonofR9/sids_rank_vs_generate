from __future__ import annotations

import numpy as np
import polars as pl

from .config import UserAwareRQKMeansConfig
from .rq_kmeans import RQKMeans
from .user_aware import (
    IUserAwareSidPipeline,
    UserAwareSidTrainingData,
    validate_user_sid_output,
)


class UserAwareRQKMeansPipeline(IUserAwareSidPipeline):
    """Квантизация признаков пар через residual K-Means"""

    def __init__(self, config: UserAwareRQKMeansConfig | None = None) -> None:
        self.config = config or UserAwareRQKMeansConfig()
        self._item_embeddings: dict[object, np.ndarray] = {}
        self._user_profiles: dict[object, np.ndarray] = {}
        self._unknown_user_profile: np.ndarray | None = None
        self._model: RQKMeans | None = None

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )

    def _load_item_embeddings(self, item_features: pl.DataFrame) -> None:
        column = self.config.embedding_column
        if column not in item_features.columns:
            raise ValueError(f"item features must contain {column!r}")
        matrix = np.asarray(item_features[column].to_list(), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError(
                "item embeddings must be a non-empty two-dimensional matrix"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("item embeddings must contain only finite values")
        matrix = self._normalize_rows(matrix)
        self._item_embeddings = {
            item_id: embedding
            for item_id, embedding in zip(
                item_features["item_id"].to_list(),
                matrix,
                strict=True,
            )
        }

    def _fit_user_profiles(self, interactions: pl.DataFrame) -> None:
        distinct_pairs = (
            interactions.select("uid", "item_id").unique().sort(["uid", "item_id"])
        )
        sums: dict[object, np.ndarray] = {}
        counts: dict[object, int] = {}
        for uid, item_id in distinct_pairs.iter_rows():
            embedding = self._item_embeddings[item_id]
            if uid not in sums:
                sums[uid] = np.zeros_like(embedding)
                counts[uid] = 0
            sums[uid] += embedding
            counts[uid] += 1

        users = sorted(sums, key=lambda value: (type(value).__name__, repr(value)))
        profile_matrix = np.stack([sums[uid] / counts[uid] for uid in users]).astype(
            np.float32
        )
        profile_matrix = self._normalize_rows(profile_matrix)
        self._user_profiles = {
            uid: profile for uid, profile in zip(users, profile_matrix, strict=True)
        }
        unknown = profile_matrix.mean(axis=0, keepdims=True)
        self._unknown_user_profile = self._normalize_rows(unknown)[0]

    def _profile(self, uid: object) -> np.ndarray:
        if self._unknown_user_profile is None:
            raise RuntimeError("train() must be called before encoding users")
        return self._user_profiles.get(uid, self._unknown_user_profile)

    def _pair_features(self, pairs: pl.DataFrame) -> np.ndarray:
        if self._model is None and not self._item_embeddings:
            raise RuntimeError("train() must be called before update()")
        if pairs.is_empty():
            raise ValueError("pairs must not be empty")
        item_vectors = []
        user_vectors = []
        for uid, item_id in pairs.select("uid", "item_id").iter_rows():
            try:
                item_vector = self._item_embeddings[item_id]
            except KeyError as error:
                raise ValueError(
                    f"item {item_id!r} is outside the fitted catalog"
                ) from error
            item_vectors.append(item_vector)
            user_vectors.append(self._profile(uid))
        items = np.stack(item_vectors).astype(np.float32)
        users = (
            np.stack(user_vectors).astype(np.float32) * self.config.user_profile_weight
        )
        features = np.concatenate(
            [items, users, items * users, np.abs(items - users)],
            axis=1,
        )
        return self._normalize_rows(features)

    def train(self, data: UserAwareSidTrainingData) -> None:
        self._load_item_embeddings(data.item_features.sort("item_id"))
        self._fit_user_profiles(data.fit_interactions)
        fit_pairs = (
            data.fit_interactions.select("uid", "item_id")
            .unique()
            .sort(["uid", "item_id"])
        )
        if self.config.codebook_sizes is not None:
            self._model = RQKMeans(
                codebook_sizes=self.config.codebook_sizes,
                kmeans_batch_size=self.config.kmeans_batch_size,
                predict_batch_size=self.config.predict_batch_size,
            )
        else:
            self._model = RQKMeans(
                num_levels=self.config.num_levels,
                codebook_size=self.config.codebook_size,
                kmeans_batch_size=self.config.kmeans_batch_size,
                predict_batch_size=self.config.predict_batch_size,
            )
        self._model.fit(self._pair_features(fit_pairs))

    def update(self, pairs: pl.DataFrame) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("train() must be called before update()")
        if "uid" not in pairs.columns or "item_id" not in pairs.columns:
            raise ValueError("pairs must contain uid and item_id")
        if pairs.select("uid", "item_id").unique().height != pairs.height:
            raise ValueError("pairs must contain unique (uid, item_id) rows")
        outputs = []
        for offset in range(0, pairs.height, self.config.predict_batch_size):
            batch = pairs.slice(offset, self.config.predict_batch_size)
            codes = self._model.get_semantic_ids(self._pair_features(batch))
            output = batch.select("uid", "item_id")
            output = output.with_columns(
                *(
                    pl.Series(
                        f"usid_{level}",
                        codes[:, level],
                        dtype=pl.Int32,
                    )
                    for level in range(codes.shape[1])
                )
            )
            outputs.append(output)
        output = pl.concat(outputs)
        validate_user_sid_output(pairs, output)
        return output

    def user_features(self, users: pl.DataFrame) -> pl.DataFrame:
        if "uid" not in users.columns:
            raise ValueError("users must contain uid")
        profiles = [self._profile(uid).tolist() for uid in users["uid"].to_list()]
        return users.select("uid").with_columns(
            pl.Series("user_feature", profiles, dtype=pl.List(pl.Float32))
        )
