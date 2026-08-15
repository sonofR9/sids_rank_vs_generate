from __future__ import annotations

import math

import torch

from sids_rank_vs_generate.diger_native_evaluation import (
    CatalogIndex,
    batch_metric_sums,
    parse_cutoffs,
)


def _codes() -> list[list[int]]:
    return [
        [-1, -1, -1, -1],
        [1, 2, 3, 0],
        [1, 2, 3, 1],
        [4, 5, 6, 0],
    ]


def test_catalog_index_builds_prefixes_and_ranked_items() -> None:
    catalog = CatalogIndex(_codes())

    assert catalog.allowed_tokens(0, torch.tensor([0])) == [1, 4]
    assert catalog.allowed_tokens(0, torch.tensor([0, 1, 2, 3])) == [0, 1]
    assert catalog.ranked_items([[1, 2, 3, 1], [1, 2, 3, 0]], 2) == [2, 1]


def test_native_batch_metrics_keep_sid_and_item_ranks_separate() -> None:
    catalog = CatalogIndex(_codes())
    outputs = torch.tensor(
        [
            [[1, 2, 3, 1], [1, 2, 3, 0], [9, 9, 9, 9]],
            [[9, 9, 9, 9], [4, 5, 6, 0], [1, 2, 3, 0]],
        ]
    )
    labels = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])

    sums, recommended = batch_metric_sums(
        outputs,
        labels,
        catalog,
        cutoffs=(1, 2, 3),
    )

    assert sums["sid_recall@1"] == 0
    assert sums["sid_recall@2"] == 2
    assert sums["item_recall@1"] == 1
    assert sums["item_recall@2"] == 2
    assert sums["item_mrr@3"] == 1.5
    assert math.isclose(sums["item_ndcg@3"], 1 + 1 / math.log2(3))
    assert sums["valid_sid_count"] == 4
    assert sums["generated_sid_count"] == 6
    assert recommended == {1, 2, 3}


def test_parse_cutoffs_deduplicates_and_sorts() -> None:
    assert parse_cutoffs("100,1,10,10") == (1, 10, 100)
