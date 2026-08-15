from __future__ import annotations

import pytest

from sids_rank_vs_generate.diger_native_matrix import (
    DEFAULT_CANDIDATES,
    select_candidate,
)


def _result(recall: float, mrr: float) -> dict[str, object]:
    return {"metrics": {"item_recall@100": recall, "item_mrr@100": mrr}}


def test_select_candidate_uses_validation_recall_then_mrr() -> None:
    results = {
        "native-b150": _result(0.10, 0.04),
        "catalog-b150": _result(0.12, 0.03),
        "catalog-b300": _result(0.12, 0.05),
    }

    selected = select_candidate(results)

    assert selected.name == "catalog-b300"


def test_select_candidate_prefers_smaller_beam_on_tie() -> None:
    results = {candidate.name: _result(0.12, 0.05) for candidate in DEFAULT_CANDIDATES}

    selected = select_candidate(results)

    assert selected.name == "native-b150"


def test_select_candidate_requires_every_validation_result() -> None:
    with pytest.raises(ValueError, match="missing DIGER validation results"):
        select_candidate({"native-b150": _result(0.10, 0.04)})
