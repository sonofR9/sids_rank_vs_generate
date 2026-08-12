from __future__ import annotations

import ast
import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from sids_rank_vs_generate.user_aware_quality_matrix import (
    BASE_CODEBOOK_SIZES,
    CAPACITY_DEPTH_SERIES,
    CAPACITY_WIDTH_SERIES,
    EXPECTED_CATALOG_SIZE,
    GLOBAL_SID_ANCHOR_ID,
    GLOBAL_SID_CODEBOOK_SIZES,
    GLOBAL_SID_REFERENCE_SCOPE,
    LEGACY_POSTER_REFERENCE_MATCH,
    MATRIX_ID,
    PREFIX_PERMUTATION_SERIES,
    PROTOCOL_ID,
    PROTOCOL_TAG,
    QUALITY_METRIC_SCHEMA,
    TAIL_CODEBOOK_SERIES,
    USER_AWARE_QUALITY_MATRIX,
    USER_WEIGHT_SERIES,
    V1_FAMILY,
    V2_FAMILY,
    _csv_text,
    main,
    matrix_manifest,
)


def _family_runs(family: str):
    return tuple(run for run in USER_AWARE_QUALITY_MATRIX if run.family == family)


def _source_constant(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"constant {name} not found in {relative_path}")


def test_matrix_contains_exactly_30_unique_runs() -> None:
    assert len(USER_AWARE_QUALITY_MATRIX) == 30
    assert len({run.run_name for run in USER_AWARE_QUALITY_MATRIX}) == 30
    assert len({run.signature for run in USER_AWARE_QUALITY_MATRIX}) == 30
    assert len(_family_runs(V1_FAMILY)) == 15
    assert len(_family_runs(V2_FAMILY)) == 15


def test_dependency_free_contract_constants_match_runtime_sources() -> None:
    assert PROTOCOL_ID == _source_constant(
        "src/sids_rank_vs_generate/data/protocols.py",
        "POSTER_FULL_33148_PROTOCOL_ID",
    )
    assert EXPECTED_CATALOG_SIZE == _source_constant(
        "src/sids_rank_vs_generate/data/protocols.py",
        "POSTER_FULL_33148_CATALOG_SIZE",
    )
    assert QUALITY_METRIC_SCHEMA == _source_constant(
        "src/sids_rank_vs_generate/evaluation/sid_quality.py",
        "SID_QUALITY_SCHEMA_VERSION",
    )


@pytest.mark.parametrize("family", (V1_FAMILY, V2_FAMILY))
def test_anchor_is_deduplicated_across_all_series(family: str) -> None:
    anchors = [
        run
        for run in _family_runs(family)
        if run.personal_codebook_sizes == BASE_CODEBOOK_SIZES
        and run.level_permutation == (0, 1, 2)
        and run.user_profile_weight == 1.0
        and (run.fusion_weight is None or run.fusion_weight == 0.35)
    ]
    assert len(anchors) == 1
    assert anchors[0].series == (
        CAPACITY_DEPTH_SERIES,
        CAPACITY_WIDTH_SERIES,
        USER_WEIGHT_SERIES,
        TAIL_CODEBOOK_SERIES,
        PREFIX_PERMUTATION_SERIES,
    )


@pytest.mark.parametrize("family", (V1_FAMILY, V2_FAMILY))
def test_capacity_tail_and_permutation_grid(family: str) -> None:
    runs = _family_runs(family)
    depth = {
        run.personal_codebook_sizes
        for run in runs
        if CAPACITY_DEPTH_SERIES in run.series
    }
    width = {
        run.personal_codebook_sizes
        for run in runs
        if CAPACITY_WIDTH_SERIES in run.series
    }
    tail = {
        run.personal_codebook_sizes
        for run in runs
        if TAIL_CODEBOOK_SERIES in run.series
    }
    permutations = {
        run.level_permutation
        for run in runs
        if PREFIX_PERMUTATION_SERIES in run.series
    }
    assert depth == {(256, 256), (256, 256, 256), (256, 256, 256, 256)}
    assert width == {
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    }
    assert tail == {
        (256, 256, 32),
        (256, 256, 64),
        (256, 256, 128),
        (256, 256, 256),
    }
    assert permutations == {(0, 1, 2), (1, 0, 2), (2, 1, 0)}


def test_weight_grids_are_exact() -> None:
    v1_weights = {
        run.user_profile_weight
        for run in _family_runs(V1_FAMILY)
        if USER_WEIGHT_SERIES in run.series
    }
    v2_weights = {
        run.fusion_weight
        for run in _family_runs(V2_FAMILY)
        if USER_WEIGHT_SERIES in run.series
    }
    assert v1_weights == {0.0, 0.25, 0.5, 1.0, 2.0}
    assert v2_weights == {0.0, 0.2, 0.35, 0.7, 1.0}


def test_every_run_has_fixed_protocol_global_sid_and_tags() -> None:
    for run in USER_AWARE_QUALITY_MATRIX:
        assert run.tags["matrix_id"] == MATRIX_ID
        assert run.tags["matrix_run_id"] == run.run_name
        assert run.tags["protocol_id"] == PROTOCOL_ID
        assert run.tags["protocol"] == PROTOCOL_TAG
        assert run.tags["quality_metric_schema"] == QUALITY_METRIC_SCHEMA
        assert run.tags["expected_catalog_size"] == str(EXPECTED_CATALOG_SIZE)
        assert run.tags["data_scope"] == "full"
        assert run.tags["evaluation_split"] == "legacy_heldout_reused"
        assert run.tags["x_scope"] == "user_catalog_macro"
        assert run.tags["quality_aggregation"] == "per_user_catalog_macro"
        assert (
            run.tags["downstream_recipe"]
            == "personalized_user_conditioned_sid_selected"
        )
        assert run.tags["pooled_global_correlation"] == "disallowed"
        assert run.tags["global_sid_codebook_sizes"] == "256,256,256"
        assert run.tags["global_sid_anchor_id"] == GLOBAL_SID_ANCHOR_ID
        assert (
            run.tags["global_sid_reference_scope"]
            == GLOBAL_SID_REFERENCE_SCOPE
        )
        assert (
            run.tags["legacy_poster_reference_match"]
            == LEGACY_POSTER_REFERENCE_MATCH
        )
        assert run.tags["downstream_metric"] == "final_item_recall_at_100"

    manifest = matrix_manifest()
    assert manifest["dry_run"] is True
    assert manifest["executes_training"] is False
    assert manifest["run_count"] == 30
    assert manifest["protocol"] == {
        "id": PROTOCOL_ID,
        "expected_catalog_size": EXPECTED_CATALOG_SIZE,
    }
    assert manifest["fixed_global_sid"]["codebook_sizes"] == list(
        GLOBAL_SID_CODEBOOK_SIZES
    )
    assert manifest["fixed_global_sid"]["anchor_id"] == GLOBAL_SID_ANCHOR_ID
    assert (
        manifest["fixed_global_sid"]["reference_scope"]
        == GLOBAL_SID_REFERENCE_SCOPE
    )
    assert (
        manifest["fixed_global_sid"]["legacy_poster_reference_match"]
        == LEGACY_POSTER_REFERENCE_MATCH
    )


def test_csv_contains_one_row_per_run() -> None:
    rows = list(csv.DictReader(StringIO(_csv_text(USER_AWARE_QUALITY_MATRIX))))
    assert len(rows) == 30
    assert {row["run_name"] for row in rows} == {
        run.run_name for run in USER_AWARE_QUALITY_MATRIX
    }
    assert {row["global_sid_anchor_id"] for row in rows} == {
        GLOBAL_SID_ANCHOR_ID
    }
    assert {row["legacy_poster_reference_match"] for row in rows} == {
        LEGACY_POSTER_REFERENCE_MATCH
    }


def test_cli_json_filter_is_still_dry_run(capsys) -> None:
    assert main(["--format", "json", "--family", V1_FAMILY]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["executes_training"] is False
    assert output["run_count"] == 15
    assert {run["family"] for run in output["runs"]} == {V1_FAMILY}
