from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import polars as pl

from .base import ISidPipeline
from .collision_resolution import get_sid_cols


@dataclass(frozen=True)
class PrecomputedSidConfig:
    path: Path
    codebook_sizes: tuple[int, ...]
    item_id_space: Literal["original", "prepared"] = "original"
    expected_catalog_size: int | None = None

    def __post_init__(self) -> None:
        if not self.codebook_sizes or any(size <= 0 for size in self.codebook_sizes):
            raise ValueError("codebook_sizes must contain positive integers")
        if self.item_id_space not in {"original", "prepared"}:
            raise ValueError("item_id_space must be original or prepared")
        if self.expected_catalog_size is not None and self.expected_catalog_size <= 0:
            raise ValueError("expected_catalog_size must be positive")


class PrecomputedSidPipeline(ISidPipeline):
    def __init__(self, config: PrecomputedSidConfig) -> None:
        self.config = config
        self._item_mapping: pl.DataFrame | None = None
        self._sids: pl.DataFrame | None = None
        self._source_sha256: str | None = None

    def set_item_mapping(self, mapping: pl.DataFrame) -> None:
        required = {"original_item_id", "item_id"}
        missing = required - set(mapping.columns)
        if missing:
            raise ValueError(
                "item mapping is missing columns: " + ", ".join(sorted(missing))
            )
        selected = mapping.select("original_item_id", "item_id")
        if selected.is_empty():
            raise ValueError("item mapping must not be empty")
        for column in required:
            if selected[column].null_count():
                raise ValueError(f"item mapping contains nulls in {column}")
            if selected[column].n_unique() != selected.height:
                raise ValueError(f"item mapping contains duplicate {column} values")
        self._item_mapping = selected

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load(self) -> pl.DataFrame:
        path = self.config.path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"precomputed SID parquet not found: {path}")
        source = pl.read_parquet(path)
        if source.is_empty():
            raise ValueError("precomputed SID parquet must not be empty")
        if "item_id" not in source.columns:
            raise ValueError("precomputed SID parquet must contain item_id")
        if source["item_id"].null_count():
            raise ValueError("precomputed SID parquet contains null item_id values")
        if source["item_id"].n_unique() != source.height:
            raise ValueError("precomputed SID parquet contains duplicate item_id values")

        sid_columns = get_sid_cols(source)
        expected_columns = [
            f"sid_{level}" for level in range(len(self.config.codebook_sizes))
        ]
        if sid_columns != expected_columns:
            raise ValueError(
                "precomputed SID levels do not match codebook_sizes: "
                f"expected {expected_columns}, received {sid_columns}"
            )
        for column, size in zip(
            sid_columns,
            self.config.codebook_sizes,
            strict=True,
        ):
            if source[column].null_count():
                raise ValueError(f"precomputed SID parquet contains nulls in {column}")
            if not source.schema[column].is_integer():
                raise TypeError(f"precomputed SID column {column} must be integer")
            minimum = int(source[column].min())
            maximum = int(source[column].max())
            if minimum < 0 or maximum >= size:
                raise ValueError(
                    f"precomputed SID column {column} is outside [0, {size})"
                )
        self._source_sha256 = self._file_sha256(path)
        return source.select("item_id", *sid_columns)

    def train(self, df: pl.DataFrame) -> None:
        source = self._load()
        if self.config.item_id_space == "original":
            if self._item_mapping is None:
                raise RuntimeError("original item IDs require the prepared item mapping")
            source = (
                source.rename({"item_id": "original_item_id"})
                .join(self._item_mapping, on="original_item_id", how="inner")
                .select("item_id", *get_sid_cols(source))
            )
        if (
            self.config.expected_catalog_size is not None
            and source.height != self.config.expected_catalog_size
        ):
            raise ValueError(
                "precomputed SID catalog size mismatch: expected "
                f"{self.config.expected_catalog_size}, received {source.height}"
            )
        if source.height != df.height:
            raise ValueError(
                "precomputed SID catalog does not cover the prepared catalog: "
                f"received {source.height} of {df.height} items"
            )
        if set(source["item_id"].to_list()) != set(df["item_id"].to_list()):
            raise ValueError("precomputed SID item IDs do not match the prepared catalog")
        self._sids = source.sort("item_id")

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._sids is None:
            raise RuntimeError("train() must be called before update()")
        if set(self._sids["item_id"].to_list()) != set(df["item_id"].to_list()):
            raise ValueError("update item IDs differ from the trained catalog")
        return self._sids

    def summary(self) -> dict[str, object]:
        return {
            "source": str(self.config.path.expanduser().resolve()),
            "source_sha256": self._source_sha256,
            "item_id_space": self.config.item_id_space,
            "item_count": self._sids.height if self._sids is not None else None,
            "sid_columns": get_sid_cols(self._sids) if self._sids is not None else None,
        }
