from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scripts.verify_global_sid_anchor import (
    CANONICAL_COLUMNS,
    CURRENT_V100_FP32_PROFILE,
    EXPECTED_CATALOG_SIZE,
    canonical_semantic_ids_sha256,
    main,
    verify,
)
from scripts.verify_poster_bridge import (
    CURRENT_V100_FP32_FIT_FINGERPRINT,
    POSTER_FULL_33148_DATA_CONTRACT,
    POSTER_FULL_33148_PROTOCOL_ID,
    POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE,
    POSTER_SID_RECIPE,
)


def _write_semantic_ids(path: Path, *, reverse: bool = True) -> None:
    item_ids = np.arange(EXPECTED_CATALOG_SIZE, dtype=np.int64)
    frame = pl.DataFrame(
        {
            "item_id": item_ids,
            "sid_0": item_ids % 256,
            "sid_1": (item_ids // 256) % 256,
            "sid_2": (item_ids * 17) % 256,
        }
    ).with_columns(pl.all().cast(pl.Int32))
    if reverse:
        frame = frame.reverse()
    frame.write_parquet(path)


def _payload(semantic_ids_path: Path) -> dict[str, Any]:
    reference = POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE
    return {
        "config": {"protocol_id": POSTER_FULL_33148_PROTOCOL_ID},
        "data": {
            **POSTER_FULL_33148_DATA_CONTRACT,
            "protocol_id": POSTER_FULL_33148_PROTOCOL_ID,
            "validation_reuses_test": True,
            "fit_fingerprint": CURRENT_V100_FP32_FIT_FINGERPRINT,
        },
        "sid": {
            **POSTER_SID_RECIPE,
            "config": {
                "num_levels": 3,
                "codebook_size": 512,
                "codebook_sizes": [256, 256, 256],
                "kmeans_batch_size": 16_384,
                "predict_batch_size": 65_536,
            },
            "statistics": {
                "independent_code_rate": reference["independent_code_rate"],
                "collision_rate": reference["collision_rate"],
                "collision_percentiles": {
                    "p95": reference["collision_p95"],
                },
                "max_collision": reference["max_collision"],
                "codebook_quality": {
                    "codebook_utilization_rate_total": reference[
                        "codebook_utilization_rate_total"
                    ],
                },
                "code_gini": reference["code_gini"],
                "code_entropy": reference["code_entropy"],
                "intra_code_similarity": reference["intra_code_similarity"],
                "prefix_coherence_at_1": reference["prefix_coherence_at_1"],
                "prefix_coherence_at_2": reference["prefix_coherence_at_2"],
                "prefix_coherence_at_3": reference["prefix_coherence_at_3"],
                "valid_sid_rate": reference["valid_sid_rate"],
            },
        },
        "artifacts": {"semantic_ids": str(semantic_ids_path)},
    }


class GlobalSidAnchorVerifierTest(unittest.TestCase):
    def test_accepts_current_anchor_without_reading_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_ids_path = root / "semantic_ids.parquet"
            metrics_path = root / "metrics.json"
            _write_semantic_ids(semantic_ids_path)
            expected = canonical_semantic_ids_sha256(semantic_ids_path)
            payload = _payload(semantic_ids_path)

            result = verify(
                payload,
                metrics_path=metrics_path,
                expected_sha256=expected.upper(),
                profile=CURRENT_V100_FP32_PROFILE,
            )

        self.assertEqual(result["canonical_sha256"], expected)
        self.assertEqual(result["canonical_columns"], list(CANONICAL_COLUMNS))
        self.assertNotIn(
            "legacy_heldout_recall_at_100", result["quality_metrics"]
        )

    def test_canonical_hash_is_independent_of_row_order_and_integer_width(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.parquet"
            second = root / "second.parquet"
            _write_semantic_ids(first, reverse=True)
            _write_semantic_ids(second, reverse=False)
            second_frame = pl.read_parquet(second).with_columns(
                pl.all().cast(pl.Int64)
            )
            second_frame.write_parquet(second)

            self.assertEqual(
                canonical_semantic_ids_sha256(first),
                canonical_semantic_ids_sha256(second),
            )

    def test_rejects_wrong_hash_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_ids_path = root / "semantic_ids.parquet"
            _write_semantic_ids(semantic_ids_path)

            with self.assertRaisesRegex(ValueError, "canonical SHA256 mismatch"):
                verify(
                    _payload(semantic_ids_path),
                    metrics_path=root / "metrics.json",
                    expected_sha256="0" * 64,
                    profile=CURRENT_V100_FP32_PROFILE,
                )

    def test_rejects_protocol_recipe_quality_and_fingerprint_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_ids_path = root / "semantic_ids.parquet"
            _write_semantic_ids(semantic_ids_path)
            expected = canonical_semantic_ids_sha256(semantic_ids_path)
            mutations = (
                (("config", "protocol_id"), "three-way"),
                (("data", "catalog_size"), EXPECTED_CATALOG_SIZE - 1),
                (("data", "fit_fingerprint"), "wrong"),
                (("sid", "class"), "NotRQKMeans"),
                (("sid", "config", "codebook_sizes"), [256, 256, 512]),
                (("sid", "statistics", "collision_rate"), 0.0),
            )
            for path, value in mutations:
                with self.subTest(path=path):
                    payload = copy.deepcopy(_payload(semantic_ids_path))
                    target = payload
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    with self.assertRaises(ValueError):
                        verify(
                            payload,
                            metrics_path=root / "metrics.json",
                            expected_sha256=expected,
                            profile=CURRENT_V100_FP32_PROFILE,
                        )

    def test_rejects_noncanonical_artifact_schema_and_duplicate_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_ids_path = root / "semantic_ids.parquet"
            _write_semantic_ids(semantic_ids_path)
            frame = pl.read_parquet(semantic_ids_path)
            frame = frame.with_columns(pl.lit(1).alias("extra"))
            frame.write_parquet(semantic_ids_path)
            with self.assertRaisesRegex(ValueError, "exactly"):
                canonical_semantic_ids_sha256(semantic_ids_path)

            _write_semantic_ids(semantic_ids_path)
            frame = pl.read_parquet(semantic_ids_path)
            frame = frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(frame["item_id"][1])
                .otherwise(pl.col("item_id"))
                .alias("item_id")
            )
            frame.write_parquet(semantic_ids_path)
            with self.assertRaisesRegex(ValueError, "duplicate item_id"):
                canonical_semantic_ids_sha256(semantic_ids_path)

    def test_cli_requires_explicit_current_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            semantic_ids_path = root / "semantic_ids.parquet"
            metrics_path = root / "metrics.json"
            _write_semantic_ids(semantic_ids_path)
            expected = canonical_semantic_ids_sha256(semantic_ids_path)
            metrics_path.write_text(
                json.dumps(_payload(semantic_ids_path)),
                encoding="utf-8",
            )

            main(
                [
                    str(metrics_path),
                    "--expected-sha256",
                    expected,
                    "--profile",
                    CURRENT_V100_FP32_PROFILE,
                ]
            )


if __name__ == "__main__":
    unittest.main()
