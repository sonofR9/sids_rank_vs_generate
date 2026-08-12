from __future__ import annotations

import copy
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from scripts.verify_poster_bridge import (
    COSINE_TOLERANCE,
    COUNT_DERIVED_TOLERANCE,
    CURRENT_V100_FP32_FIT_FINGERPRINT,
    CURRENT_V100_FP32_PROFILE,
    DEFAULT_REFERENCE_PROFILE,
    LEGACY_HISTORICAL_PROFILE,
    POSTER_FULL_33148_DATA_CONTRACT,
    POSTER_FULL_33148_PROTOCOL_ID,
    POSTER_GENERATION_EXAMPLES,
    POSTER_GENERATION_RECIPE,
    POSTER_RQKMEANS_L3_C256_REFERENCE,
    POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE,
    POSTER_SID_CONFIG,
    POSTER_SID_RECIPE,
    RECALL_TOLERANCE,
    main,
    verify,
)


def valid_payload(
    reference_profile: str = DEFAULT_REFERENCE_PROFILE,
) -> dict[str, Any]:
    reference = (
        POSTER_RQKMEANS_L3_C256_REFERENCE
        if reference_profile == LEGACY_HISTORICAL_PROFILE
        else POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE
    )
    recall = reference["legacy_heldout_recall_at_100"]
    return {
        "config": {
            "dataset_size": "50m",
            "core_min_interactions_per_item": 5,
            "test_days": 7.0,
            "timestamp_unit_seconds": 5.0,
            "catalog_from_fit": False,
            "latest_interactions": None,
            "enabled_stages": ["generation"],
            "protocol_id": POSTER_FULL_33148_PROTOCOL_ID,
            "expected_catalog_size": 33_148,
        },
        "data": {
            **POSTER_FULL_33148_DATA_CONTRACT,
            "test_start_timestamp": "25878770.0",
            "validation_start_timestamp": "25878770.0",
            "protocol_id": POSTER_FULL_33148_PROTOCOL_ID,
            "validation_reuses_test": True,
            "fit_fingerprint": CURRENT_V100_FP32_FIT_FINGERPRINT,
        },
        "sid": {
            **POSTER_SID_RECIPE,
            "config": dict(POSTER_SID_CONFIG),
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
        "user_sid": None,
        "evaluations": {
            "generation": {
                "target_sid_type": "global",
                "examples": dict(POSTER_GENERATION_EXAMPLES),
                "config": dict(POSTER_GENERATION_RECIPE),
                "validation_metrics": {"recall@100": recall},
                "metrics": {"recall@100": recall},
                "loaded_checkpoint": None,
            },
        },
    }


class PosterBridgeVerifierTest(unittest.TestCase):
    def test_preserves_the_legacy_historical_reference_by_default(self) -> None:
        result = verify(valid_payload())

        self.assertEqual(
            result["legacy_heldout_recall_at_100"],
            POSTER_RQKMEANS_L3_C256_REFERENCE[
                "legacy_heldout_recall_at_100"
            ],
        )

    def test_accepts_the_explicit_v100_fp32_profile(self) -> None:
        result = verify(
            valid_payload(CURRENT_V100_FP32_PROFILE),
            reference_profile=CURRENT_V100_FP32_PROFILE,
        )

        self.assertEqual(
            result["legacy_heldout_recall_at_100"],
            POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE[
                "legacy_heldout_recall_at_100"
            ],
        )

    def test_cli_prints_the_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(
                json.dumps(valid_payload(CURRENT_V100_FP32_PROFILE)),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        str(metrics_path),
                        "--profile",
                        CURRENT_V100_FP32_PROFILE,
                    ]
                )

        result = json.loads(output.getvalue())
        self.assertEqual(result["reference_profile"], CURRENT_V100_FP32_PROFILE)
        self.assertEqual(
            result["metrics"]["independent_code_rate"],
            POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE[
                "independent_code_rate"
            ],
        )

    def test_does_not_mix_reference_profiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "independent_code_rate"):
            verify(
                valid_payload(LEGACY_HISTORICAL_PROFILE),
                reference_profile=CURRENT_V100_FP32_PROFILE,
            )

    def test_current_profile_requires_the_fit_fingerprint(self) -> None:
        payload = valid_payload(CURRENT_V100_FP32_PROFILE)
        payload["data"]["fit_fingerprint"] = "wrong"

        with self.assertRaisesRegex(ValueError, "fit_fingerprint"):
            verify(payload, reference_profile=CURRENT_V100_FP32_PROFILE)

    def test_rejects_forged_downstream_recipe(self) -> None:
        mutations = (
            (("sid", "class"), "NotRQKMeans"),
            (("evaluations", "generation", "target_sid_type"), "personalized"),
            (("evaluations", "generation", "examples", "train"), 1),
            (("evaluations", "generation", "config", "max_history_len"), 999),
            (("evaluations", "generation", "config", "num_beams"), 1),
            (("evaluations", "generation", "config", "resolve_collisions"), True),
            (("evaluations", "generation", "config", "use_bf16"), True),
            (("evaluations", "generation", "loaded_checkpoint"), "old.pt"),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(valid_payload())
                target = payload
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value

                with self.assertRaises(ValueError):
                    verify(payload)

    def test_uses_metric_specific_tolerances(self) -> None:
        accepted = valid_payload()
        accepted["sid"]["statistics"]["intra_code_similarity"] += (
            COSINE_TOLERANCE * 0.5
        )
        accepted["evaluations"]["generation"]["metrics"]["recall@100"] += (
            RECALL_TOLERANCE * 0.5
        )
        accepted["evaluations"]["generation"]["validation_metrics"][
            "recall@100"
        ] = accepted["evaluations"]["generation"]["metrics"]["recall@100"]
        verify(accepted)

        count_mismatch = valid_payload()
        count_mismatch["sid"]["statistics"]["code_entropy"] += (
            COUNT_DERIVED_TOLERANCE * 2
        )
        with self.assertRaisesRegex(ValueError, "code_entropy"):
            verify(count_mismatch)

        cosine_mismatch = valid_payload()
        cosine_mismatch["sid"]["statistics"]["intra_code_similarity"] += (
            COSINE_TOLERANCE * 2
        )
        with self.assertRaisesRegex(ValueError, "intra_code_similarity"):
            verify(cosine_mismatch)

        recall_mismatch = valid_payload()
        recall_mismatch["evaluations"]["generation"]["metrics"][
            "recall@100"
        ] += RECALL_TOLERANCE * 2
        recall_mismatch["evaluations"]["generation"]["validation_metrics"][
            "recall@100"
        ] = recall_mismatch["evaluations"]["generation"]["metrics"][
            "recall@100"
        ]
        with self.assertRaisesRegex(ValueError, "legacy_heldout_recall_at_100"):
            verify(recall_mismatch)

    def test_rejects_non_finite_metrics(self) -> None:
        payload = valid_payload()
        payload["sid"]["statistics"]["collision_rate"] = math.nan

        with self.assertRaisesRegex(ValueError, "not finite"):
            verify(payload)


if __name__ == "__main__":
    unittest.main()
