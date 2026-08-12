from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from sids_rank_vs_generate.data.protocols import (
    POSTER_FULL_33148_PROTOCOL_ID,
)
from sids_rank_vs_generate.sids.level_permutation import (
    LevelPermutationUserAwareSidPipeline,
)
from sids_rank_vs_generate.sids.ranker_embedding_user_aware import (
    RankerEmbeddingUserAwareRQKMeansPipeline,
)
from sids_rank_vs_generate.sids.user_aware import (
    IUserAwareSidPipeline,
    UserAwareSidTrainingData,
)
from sids_rank_vs_generate.sids.user_aware_rq_kmeans import (
    UserAwareRQKMeansPipeline,
)
from sids_rank_vs_generate.user_aware_quality_experiment import (
    FIXED_GENERATION_CONFIG,
    FIXED_MAPPER_CONFIG,
    build_user_aware_quality_experiment_by_name,
    get_user_aware_quality_run,
    validate_user_aware_quality_experiment,
)
from sids_rank_vs_generate.user_aware_quality_matrix import (
    GLOBAL_SID_CODEBOOK_SIZES,
    USER_AWARE_QUALITY_MATRIX,
)


class _FixedSidPipeline(IUserAwareSidPipeline):
    def train(self, data: UserAwareSidTrainingData) -> None:
        del data

    def update(self, pairs: pl.DataFrame) -> pl.DataFrame:
        return pairs.with_columns(
            pl.Series("usid_0", [10, 11], dtype=pl.Int32),
            pl.Series("usid_1", [20, 21], dtype=pl.Int32),
            pl.Series("usid_2", [30, 31], dtype=pl.Int32),
        )

    def user_features(self, users: pl.DataFrame) -> pl.DataFrame:
        return users.with_columns(
            pl.Series("user_feature", [[1.0]] * users.height)
        )


def test_level_permutation_only_reorders_assigned_codes() -> None:
    pairs = pl.DataFrame({"uid": [1, 2], "item_id": [100, 200]})
    wrapper = LevelPermutationUserAwareSidPipeline(
        _FixedSidPipeline(),
        codebook_sizes=(32, 64, 128),
        level_permutation=(2, 1, 0),
    )

    result = wrapper.update(pairs)

    assert result.to_dict(as_series=False) == {
        "uid": [1, 2],
        "item_id": [100, 200],
        "usid_0": [30, 31],
        "usid_1": [20, 21],
        "usid_2": [10, 11],
    }
    assert wrapper.config.codebook_sizes == (128, 64, 32)
    assert wrapper.pipeline.__class__ is _FixedSidPipeline


@pytest.mark.parametrize(
    ("run_name", "expected_type"),
    (
        ("uaq30-v1-anchor-l3-c256", UserAwareRQKMeansPipeline),
        (
            "uaq30-v2-anchor-l3-c256",
            RankerEmbeddingUserAwareRQKMeansPipeline,
        ),
    ),
)
def test_matrix_row_builds_common_legacy_experiment(
    run_name: str,
    expected_type: type[IUserAwareSidPipeline],
) -> None:
    config = build_user_aware_quality_experiment_by_name(run_name)
    generation = config.training.generation

    assert config.dataset.protocol_id == POSTER_FULL_33148_PROTOCOL_ID
    assert config.dataset.history_days is None
    assert config.catalog_from_fit is False
    assert config.sid_pipeline.config.codebook_sizes == GLOBAL_SID_CODEBOOK_SIZES
    assert isinstance(config.user_sid_pipeline, expected_type)
    assert config.user_catalog_quality.enabled is True
    assert config.user_catalog_quality.max_users == 256
    assert config.user_catalog_quality.selection == "stable_hash"
    assert config.user_catalog_quality.seed == 42
    assert generation.max_history_len == 16
    assert generation.num_beams == 150
    assert generation.recall_k == 100
    assert generation.constrain_to_catalog is True
    assert generation.resolve_collisions is False
    assert generation.max_train_examples is None
    assert generation.max_validation_examples is None
    assert generation.max_test_examples is None
    assert generation.use_bf16 is False
    assert generation.initialize_item_embeddings_from_content is False
    assert generation.auxiliary_item_loss_weight == 0.0
    assert generation.run_diagnostics is False
    assert generation.mapper == FIXED_MAPPER_CONFIG
    assert generation.mapper.collision_level_loss_weight is None
    assert dict(config.tracking.tags)["data_scope"] == "full"
    assert (
        dict(config.tracking.tags)["evaluation_split"]
        == "legacy_heldout_reused"
    )


def test_permutation_row_builds_output_only_wrapper() -> None:
    config = build_user_aware_quality_experiment_by_name(
        "uaq30-v1-prefix-permutation-210"
    )

    assert isinstance(
        config.user_sid_pipeline,
        LevelPermutationUserAwareSidPipeline,
    )
    assert isinstance(
        config.user_sid_pipeline.pipeline,
        UserAwareRQKMeansPipeline,
    )
    assert config.user_sid_pipeline.config.level_permutation == (2, 1, 0)


def test_builder_rejects_modified_declared_row() -> None:
    run = USER_AWARE_QUALITY_MATRIX[0]
    modified = replace(run, user_profile_weight=99.0)

    from sids_rank_vs_generate.user_aware_quality_experiment import (
        build_user_aware_quality_experiment,
    )

    with pytest.raises(ValueError, match="declared matrix row"):
        build_user_aware_quality_experiment(modified)


def test_runtime_validation_rejects_generation_recipe_drift() -> None:
    run_name = "uaq30-v2-anchor-l3-c256"
    config = build_user_aware_quality_experiment_by_name(run_name)
    changed_generation = replace(FIXED_GENERATION_CONFIG, num_beams=149)
    changed = replace(
        config,
        training=replace(config.training, generation=changed_generation),
    )

    with pytest.raises(ValueError, match="generation config"):
        validate_user_aware_quality_experiment(
            changed,
            get_user_aware_quality_run(run_name),
        )


def test_runtime_validation_rejects_matrix_row_drift() -> None:
    config = build_user_aware_quality_experiment_by_name(
        "uaq30-v1-anchor-l3-c256"
    )

    with pytest.raises(ValueError, match="user SID config"):
        validate_user_aware_quality_experiment(
            config,
            get_user_aware_quality_run("uaq30-v1-user-weight-0p5"),
        )
