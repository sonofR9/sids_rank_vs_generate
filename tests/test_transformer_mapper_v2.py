from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl
import torch
import torch.nn.functional as F

from sids_rank_vs_generate.sids.transformer_user_sid_mapper import (
    MAPPER_CHECKPOINT_VERSION,
    TransformerUserSidMapper,
    UserSidMapperConfig,
)


def _base_config(**updates) -> UserSidMapperConfig:
    config = UserSidMapperConfig(
        kind="transformer",
        d_model=8,
        d_ff=16,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        batch_size=4,
        inference_batch_size=4,
        max_epochs=1,
        early_stopping_patience=1,
    )
    return replace(config, **updates)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1, 2],
            "user_feature": [[1.0, 0.0], [0.0, 1.0]],
        }
    )


def _training_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uid": [1, 1, 2, 2],
            "usid_0": [0, 1, 0, 1],
            "usid_1": [0, 1, 1, 0],
            "sid_0": [0, 1, 0, 1],
            "sid_1": [0, 1, 1, 1],
        }
    )


def _mapper(
    root: Path,
    config: UserSidMapperConfig,
    *,
    valid_paths=((0, 0), (0, 1), (1, 1)),
    content_sid_levels=None,
) -> TransformerUserSidMapper:
    return TransformerUserSidMapper(
        config,
        user_sid_sizes=(2, 2),
        global_sid_sizes=(2, 2),
        valid_global_sids=valid_paths,
        user_features=_features(),
        device="cpu",
        checkpoint_path=root / "mapper.pt",
        content_sid_levels=content_sid_levels,
    )


class TransformerMapperV2Test(unittest.TestCase):
    def test_defaults_keep_legacy_training_and_decoding(self):
        config = UserSidMapperConfig()

        self.assertFalse(config.catalog_masked_teacher_forced_loss)
        self.assertIsNone(config.level_loss_weights)
        self.assertIsNone(config.collision_level_loss_weight)
        self.assertEqual(config.validation_selection_metric, "teacher_forced_loss")
        self.assertIsNone(config.decode_beam_size)

    def test_catalog_mask_is_applied_before_cross_entropy_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            masked = _mapper(
                root,
                _base_config(catalog_masked_teacher_forced_loss=True),
                valid_paths=((0, 0),),
            )
            legacy = _mapper(
                root,
                _base_config(),
                valid_paths=((0, 0),),
            )
            features = torch.tensor([[1.0, 0.0]])
            user_sids = torch.tensor([[0, 0]])
            global_sids = torch.tensor([[0, 0]])

            def output(mapper):
                memory = torch.zeros(1, 1, mapper.config.d_model)
                logits = [
                    torch.tensor([[0.0, 10.0]], requires_grad=True),
                    torch.tensor([[0.0, 10.0]], requires_grad=True),
                ]
                with (
                    patch.object(
                        mapper._model,
                        "encode",
                        return_value=memory,
                    ),
                    patch.object(
                        mapper._model,
                        "decode_next",
                        side_effect=logits,
                    ),
                ):
                    return mapper._model_output(
                        features,
                        user_sids,
                        global_sids,
                    )

            masked_output = output(masked)
            legacy_output = output(legacy)

            self.assertAlmostEqual(float(masked_output["loss"].item()), 0.0)
            self.assertGreater(float(legacy_output["loss"].item()), 9.0)
            self.assertEqual(masked_output["preds"].tolist(), [[0, 0]])
            self.assertEqual(legacy_output["preds"].tolist(), [[1, 1]])

    def test_catalog_mask_is_applied_before_inference_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapper = _mapper(
                Path(tmpdir),
                _base_config(
                    validation_selection_top_k=1,
                    decode_beam_size=2,
                ),
                valid_paths=((0, 0), (1, 1)),
            )
            mapper._is_fitted = True
            with (
                patch.object(
                    mapper._model,
                    "encode",
                    return_value=torch.zeros(1, 1, mapper.config.d_model),
                ),
                patch.object(
                    mapper._model,
                    "decode_next",
                    side_effect=[
                        torch.tensor([[1.0, 0.0]]),
                        torch.tensor([[0.0, 100.0], [0.0, 0.0]]),
                    ],
                ),
            ):
                predictions = mapper.predict_topk_batch([(1, [0, 0])], k=1)

            self.assertEqual(predictions, [[(0, 0)]])

    def test_catalog_tensor_index_supports_mixed_radix_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapper = TransformerUserSidMapper(
                _base_config(),
                user_sid_sizes=(2, 2),
                global_sid_sizes=(2, 3, 4),
                valid_global_sids=((0, 2, 3), (1, 0, 1)),
                user_features=_features(),
                device="cpu",
                checkpoint_path=Path(tmpdir) / "mapper.pt",
            )

            allowed = mapper._catalog_allowed_next(
                torch.tensor([[0, 2], [1, 0]]),
                level=2,
            )

            self.assertEqual(
                allowed.tolist(),
                [
                    [False, False, False, True],
                    [False, True, False, False],
                ],
            )

    def test_collision_weight_uses_explicit_content_level_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapper = _mapper(
                Path(tmpdir),
                _base_config(collision_level_loss_weight=0.5),
                content_sid_levels=1,
            )
            features = torch.tensor([[1.0, 0.0]])
            user_sids = torch.tensor([[0, 0]])
            global_sids = torch.tensor([[0, 0]])
            level_logits = [
                torch.tensor([[10.0, 0.0]], requires_grad=True),
                torch.tensor([[0.0, 10.0]], requires_grad=True),
            ]
            with (
                patch.object(
                    mapper._model,
                    "encode",
                    return_value=torch.zeros(1, 1, mapper.config.d_model),
                ),
                patch.object(
                    mapper._model,
                    "decode_next",
                    side_effect=level_logits,
                ),
            ):
                output = mapper._model_output(features, user_sids, global_sids)

            first = F.cross_entropy(level_logits[0], global_sids[:, 0])
            collision = F.cross_entropy(level_logits[1], global_sids[:, 1])
            expected = (first + 0.5 * collision) / 1.5
            self.assertEqual(mapper._level_loss_weights, (1.0, 0.5))
            self.assertTrue(torch.allclose(output["loss"], expected))

            with self.assertRaisesRegex(ValueError, "content SID levels"):
                _mapper(
                    Path(tmpdir),
                    _base_config(collision_level_loss_weight=0.5),
                )

    def test_return_k_is_separate_from_internal_decode_beam(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapper = _mapper(
                Path(tmpdir),
                _base_config(decode_beam_size=16),
            )
            mapper._is_fitted = True
            with patch.object(
                mapper,
                "_predict_chunk",
                return_value=[[(0, 0), (0, 1), (1, 1)]],
            ) as predict_chunk:
                predictions = mapper.predict_topk_batch([(1, [0, 0])], k=3)

            self.assertEqual(predictions, [[(0, 0), (0, 1), (1, 1)]])
            self.assertEqual(predict_chunk.call_args.kwargs["return_k"], 3)
            self.assertEqual(predict_chunk.call_args.kwargs["beam_size"], 16)

    def test_checkpoint_selection_uses_autoregressive_catalog_top_k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mapper = _mapper(
                root,
                _base_config(
                    max_epochs=3,
                    early_stopping_patience=3,
                    catalog_masked_teacher_forced_loss=True,
                    collision_level_loss_weight=0.5,
                    validation_selection_metric=(
                        "autoregressive_catalog_top_k_accuracy"
                    ),
                    validation_selection_top_k=3,
                    decode_beam_size=4,
                ),
                content_sid_levels=1,
            )
            validation = [{"uid": 1, "sid": [0, 0], "global_sid": [0, 0]}]
            with (
                patch.object(
                    mapper,
                    "_loss",
                    side_effect=[0.1, 0.2, 0.05],
                ),
                patch.object(
                    mapper,
                    "_validation_catalog_top_k_accuracy",
                    side_effect=[0.2, 0.8, 0.3],
                ) as top_k_accuracy,
            ):
                mapper.fit(_training_frame(), validation)

            checkpoint = torch.load(
                root / "mapper.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(top_k_accuracy.call_count, 3)
            self.assertEqual(
                mapper.training_summary["best_validation_selection_value"],
                0.8,
            )
            self.assertEqual(mapper.training_summary["best_validation_loss"], 0.2)
            self.assertEqual(
                checkpoint["checkpoint_version"],
                MAPPER_CHECKPOINT_VERSION,
            )
            self.assertEqual(checkpoint["best_validation_selection_value"], 0.8)
            self.assertEqual(
                checkpoint["provenance"]["effective_level_loss_weights"],
                (1.0, 0.5),
            )

    def test_v2_config_rejects_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_config = _base_config()
            legacy_mapper = _mapper(root, legacy_config)
            legacy_checkpoint = root / "legacy.pt"
            torch.save(
                {
                    "model_state_dict": legacy_mapper._model.state_dict(),
                    "config": asdict(legacy_config),
                    "user_sid_sizes": (2, 2),
                    "global_sid_sizes": (2, 2),
                    "best_validation_loss": 0.5,
                },
                legacy_checkpoint,
            )

            legacy_mapper.load(_training_frame(), legacy_checkpoint)
            v2_mapper = _mapper(
                root,
                _base_config(catalog_masked_teacher_forced_loss=True),
            )
            with self.assertRaisesRegex(ValueError, "version 2 checkpoint"):
                v2_mapper.load(_training_frame(), legacy_checkpoint)

    def test_v2_checkpoint_rejects_legacy_behavior_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v2_config = _base_config(
                catalog_masked_teacher_forced_loss=True,
                collision_level_loss_weight=0.5,
                validation_selection_metric=("autoregressive_catalog_top_k_accuracy"),
                validation_selection_top_k=1,
                decode_beam_size=2,
            )
            v2_mapper = _mapper(
                root,
                v2_config,
                content_sid_levels=1,
            )
            v2_mapper.fit(
                _training_frame(),
                [{"uid": 1, "sid": [0, 0], "global_sid": [0, 0]}],
            )
            legacy_mapper = _mapper(root, _base_config())

            with self.assertRaisesRegex(ValueError, "v2 config field"):
                legacy_mapper.load(_training_frame(), root / "mapper.pt")

    def test_v2_checkpoint_rejects_catalog_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _base_config(catalog_masked_teacher_forced_loss=True)
            source_mapper = _mapper(root, config)
            source_mapper.fit(
                _training_frame(),
                [{"uid": 1, "sid": [0, 0], "global_sid": [0, 0]}],
            )
            other_catalog_mapper = _mapper(
                root,
                config,
                valid_paths=((0, 0), (0, 1), (1, 0), (1, 1)),
            )

            with self.assertRaisesRegex(ValueError, "catalog fingerprint mismatch"):
                other_catalog_mapper.load(_training_frame(), root / "mapper.pt")

    def test_loss_weights_must_be_finite(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "loss weight"):
                    _base_config(level_loss_weights=(1.0, value))
                with self.assertRaisesRegex(ValueError, "collision loss weight"):
                    _base_config(collision_level_loss_weight=value)


if __name__ == "__main__":
    unittest.main()
