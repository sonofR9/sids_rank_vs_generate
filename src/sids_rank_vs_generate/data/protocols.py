from __future__ import annotations


THREE_WAY_PROTOCOL_ID = "three-way"
POSTER_FULL_33148_PROTOCOL_ID = "poster-full-33148-v1"
POSTER_FULL_33148_CATALOG_SIZE = 33_148
POSTER_FULL_33148_DATA_CONTRACT = {
    "interactions": 634_642,
    "train_interactions": 630_095,
    "fit_interactions": 630_095,
    "validation_interactions": 4_547,
    "test_interactions": 4_547,
    "users": 8_032,
    "catalog_size": POSTER_FULL_33148_CATALOG_SIZE,
    "test_start_timestamp": 25_878_770.0,
    "validation_start_timestamp": 25_878_770.0,
}
POSTER_RQKMEANS_L3_C256_REFERENCE = {
    "independent_code_rate": 0.8627971521660432,
    "collision_rate": 0.13720284783395675,
    "collision_p95": 5.0,
    "max_collision": 28,
    "codebook_utilization_rate_total": 1.0,
    "code_gini": 0.12706173879215976,
    "code_entropy": 10.174084041926339,
    "intra_code_similarity": 0.9449189484497289,
    "prefix_coherence_at_1": 0.7103009254205972,
    "prefix_coherence_at_2": 0.8610243591478416,
    "prefix_coherence_at_3": 0.9449189484497289,
    "valid_sid_rate": 1.0,
    "legacy_heldout_recall_at_100": 0.09143621766280106,
}
POSTER_RQKMEANS_L3_C256_V100_FP32_REFERENCE = {
    "independent_code_rate": 0.8640943646675516,
    "collision_rate": 0.13590563533244837,
    "collision_p95": 5.0,
    "max_collision": 22,
    "codebook_utilization_rate_total": 1.0,
    "code_gini": 0.12609535684607587,
    "code_entropy": 10.17628737954354,
    "intra_code_similarity": 0.9452018556661516,
    "prefix_coherence_at_1": 0.7072719681937971,
    "prefix_coherence_at_2": 0.8605393418109137,
    "prefix_coherence_at_3": 0.9452018556661516,
    "valid_sid_rate": 1.0,
    "legacy_heldout_recall_at_100": 0.08563782337198929,
}

SUPPORTED_PROTOCOL_IDS = frozenset(
    {
        THREE_WAY_PROTOCOL_ID,
        POSTER_FULL_33148_PROTOCOL_ID,
    }
)


def validate_protocol_id(protocol_id: str) -> None:
    if protocol_id not in SUPPORTED_PROTOCOL_IDS:
        supported = ", ".join(sorted(SUPPORTED_PROTOCOL_IDS))
        raise ValueError(
            f"unknown data protocol {protocol_id!r}; expected one of: {supported}"
        )


def uses_legacy_heldout(protocol_id: str) -> bool:
    validate_protocol_id(protocol_id)
    return protocol_id == POSTER_FULL_33148_PROTOCOL_ID


def expected_catalog_size(protocol_id: str) -> int | None:
    validate_protocol_id(protocol_id)
    if protocol_id == POSTER_FULL_33148_PROTOCOL_ID:
        return POSTER_FULL_33148_CATALOG_SIZE
    return None


def validate_poster_full_33148_summary(summary: dict[str, object]) -> None:
    """Проверить контрольные числа старого poster-набора"""

    for key, expected in POSTER_FULL_33148_DATA_CONTRACT.items():
        if key not in summary:
            raise ValueError(f"poster protocol summary is missing {key}")
        actual = summary[key]
        if isinstance(expected, float):
            try:
                matches = float(actual) == expected
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"poster protocol data mismatch for {key}: "
                f"expected {expected}, received {actual}"
            )
