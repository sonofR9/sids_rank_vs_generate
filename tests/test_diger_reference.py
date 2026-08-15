from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from sids_rank_vs_generate.data.preprocessing import PreparedInteractions
from sids_rank_vs_generate.diger_reference import (
    DigerReferenceConfig,
    DigerReferencePaths,
    build_reference_examples,
    build_upstream_config,
    export_reference_sids,
    finalize_existing_reference_run,
    parse_reference_metrics,
)
from sids_rank_vs_generate.tracking import MLflowConfig


def _prepared() -> PreparedInteractions:
    data = pl.DataFrame(
        {
            "uid": [1, 1, 1, 1, 2, 2, 2, 2],
            "item_id": [0, 1, 2, 3, 1, 2, 3, 0],
            "timestamp": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    mapping = pl.DataFrame(
        {"original_item_id": [10, 11, 12, 13], "item_id": [0, 1, 2, 3]}
    )
    return PreparedInteractions(
        data=data,
        train=data.filter(pl.col("timestamp") < 4),
        test=data.filter(pl.col("timestamp") >= 4),
        item_mapping=mapping,
        catalog_size=4,
        test_start_timestamp=4,
        timestamp_unit_seconds=1.0,
        fit=data.filter(pl.col("timestamp") < 3),
        validation=data.filter((pl.col("timestamp") >= 3) & (pl.col("timestamp") < 4)),
        validation_start_timestamp=3,
        protocol_id="three-way",
    )


def test_reference_examples_preserve_temporal_split() -> None:
    examples = build_reference_examples(
        _prepared(), min_history_len=1, max_history_len=2
    )

    assert {name: len(rows) for name, rows in examples.items()} == {
        "train": 2,
        "valid": 2,
        "test": 2,
    }
    assert examples["test"][0] == {
        "inter_history": ["item_1", "item_2"],
        "target_id": "item_3",
    }


def test_upstream_config_uses_three_codebooks_and_conflict_token(
    tmp_path: Path,
) -> None:
    config = DigerReferenceConfig(
        variant="frqud", rqvae_epochs=1, joint_epochs=1, num_workers=1
    )
    values = build_upstream_config(
        config,
        prepared_dir=tmp_path / "dataset" / "yambda50m",
        run_dir=tmp_path / "run",
        rqvae_checkpoint=tmp_path / "rqvae.pt",
        embedding_dimension=128,
    )

    assert values["num_emb_list"] == [256, 256, 256]
    assert values["code_length"] == 4
    assert values["semantic_hidden_size"] == 128
    assert values["use_adaptive_selection"] is True
    assert values["use_learnable_sigma_gumbel"] is False
    assert values["max_length"] == config.max_history_len + 1


def test_run_name_cannot_escape_work_root(tmp_path: Path) -> None:
    try:
        DigerReferencePaths.build(tmp_path, "../outside")
    except ValueError as error:
        assert "run_name" in str(error)
    else:
        raise AssertionError("unsafe run name was accepted")


def test_parse_and_export_reference_results(tmp_path: Path) -> None:
    log_path = tmp_path / "diger.log"
    log_path.write_text(
        "Best Validation Score: 0.123\n"
        "Test Results: {'recall@1': np.float64(0.1), "
        "'recall@10': np.float32(0.2), 'ndcg@10': 0.15}\n",
        encoding="utf-8",
    )
    metrics = parse_reference_metrics(log_path)
    assert metrics == {
        "validation.best_score": 0.123,
        "test.recall@1": 0.1,
        "test.recall@10": 0.2,
        "test.ndcg@10": 0.15,
    }

    code_path = tmp_path / "best.code.json"
    code_path.write_text(
        json.dumps([[-1, -1, -1, -1], [1, 2, 3, 0], [4, 5, 6, 1]]),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "item_mapping.parquet"
    pl.DataFrame({"original_item_id": [100, 101], "item_id": [0, 1]}).write_parquet(
        mapping_path
    )
    paths = export_reference_sids(
        code_path,
        catalog_size=2,
        item_mapping_path=mapping_path,
        destination=tmp_path / "export",
        num_levels=3,
    )

    raw = pl.read_parquet(paths["raw"])
    with_conflict = pl.read_parquet(paths["with_conflict"])
    original = pl.read_parquet(paths["original_items"])
    assert raw.columns == ["item_id", "sid_0", "sid_1", "sid_2"]
    assert with_conflict.columns == ["item_id", "sid_0", "sid_1", "sid_2", "sid_3"]
    assert original["item_id"].to_list() == [100, 101]


def test_finalize_existing_run_without_retraining(tmp_path: Path) -> None:
    config = DigerReferenceConfig(
        variant="frqud", rqvae_epochs=1, joint_epochs=1, num_workers=1
    )
    paths = DigerReferencePaths.build(tmp_path, "recovered")
    paths.prepared.mkdir(parents=True)
    paths.rqvae.mkdir(parents=True)
    paths.run.mkdir(parents=True)
    (paths.prepared / "manifest.json").write_text(
        json.dumps({"config": asdict(config), "embedding_shape": [2, 3]}),
        encoding="utf-8",
    )
    pl.DataFrame({"original_item_id": [100, 101], "item_id": [0, 1]}).write_parquet(
        paths.prepared / "item_mapping.parquet"
    )
    (paths.rqvae / "best_collision_model.pth").touch()
    (paths.run / "diger_config.json").write_text("{}\n", encoding="utf-8")
    (paths.run / "diger.log").write_text(
        "Best Validation Score: 0.12\n"
        "Test Results: {'recall@10': np.float64(0.2)}\n",
        encoding="utf-8",
    )
    code_dir = paths.run / "myckpt" / "yambda50m" / "checkpoint"
    code_dir.mkdir(parents=True)
    (code_dir / "best.code.json").write_text(
        json.dumps([[-1, -1, -1, -1], [1, 2, 3, 0], [4, 5, 6, 1]]),
        encoding="utf-8",
    )

    result = finalize_existing_reference_run(
        config=config,
        paths=paths,
        tracking=MLflowConfig(enabled=False),
    )

    assert result["status"] == "FINISHED"
    assert result["metrics"]["test.recall@10"] == 0.2
    assert (paths.run / "run_complete.json").is_file()
    assert (paths.run / "sid_mapping" / "semantic_ids.parquet").is_file()
