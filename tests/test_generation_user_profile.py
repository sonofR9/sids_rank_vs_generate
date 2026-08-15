from types import SimpleNamespace
import unittest
from unittest.mock import patch

import polars as pl
import torch

from sids_rank_vs_generate.data.generation import (
    SIDLogQTable,
    UserProfileTable,
    collate_generation,
)
from sids_rank_vs_generate.evaluation.beam_search import semantic_beam_search
from sids_rank_vs_generate.evaluation.generation_stage import (
    _initial_item_embeddings,
)
from sids_rank_vs_generate.models.semantic_transformer_model import (
    SemanticTransformerModel,
)


def model_config(*, use_user_profile: bool) -> SimpleNamespace:
    return SimpleNamespace(
        num_levels=2,
        rq_num_levels=2,
        codebook_sizes=(2, 2),
        codebook_size=2,
        d_model=4,
        d_ff=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        max_history_len=4,
        semantic_history_weight=0.0,
        auxiliary_item_loss_weight=0.1,
        ranking_temperature=0.07,
        use_user_profile=use_user_profile,
        user_profile_dim=2 if use_user_profile else 0,
    )


def example(uid: int, history: list[int]) -> dict:
    return {
        "uid": uid,
        "history": history,
        "target": 2,
        "sid": [0, 1],
        "global_sid": [0, 1],
    }


class UserProfileTableTest(unittest.TestCase):
    def test_collator_looks_up_profiles_without_copying_into_examples(self):
        table = UserProfileTable.from_frame(
            pl.DataFrame(
                {
                    "uid": [2, 1],
                    "user_feature": [[0.0, 1.0], [1.0, 0.0]],
                }
            )
        )
        batch = [example(1, [1]), example(2, [1, 2])]

        collated = collate_generation(
            batch,
            num_levels=2,
            user_profiles=table,
        )
        legacy = collate_generation(batch, num_levels=2)

        self.assertTrue(
            torch.equal(
                collated["user_features"],
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            )
        )
        self.assertNotIn("user_features", legacy)
        self.assertNotIn("user_features", batch[0])

    def test_lookup_rejects_unknown_user(self):
        table = UserProfileTable.from_frame(
            pl.DataFrame({"uid": [1], "user_feature": [[1.0, 0.0]]})
        )

        with self.assertRaisesRegex(ValueError, "whole batch"):
            table.lookup([999])

    def test_profiles_and_sid_logq_can_share_one_batch(self):
        examples = [example(1, [1]), example(2, [1, 2])]
        profiles = UserProfileTable.from_frame(
            pl.DataFrame(
                {
                    "uid": [1, 2],
                    "user_feature": [[1.0, 0.0], [0.0, 1.0]],
                }
            )
        )
        sid_logq = SIDLogQTable(
            examples,
            codebook_sizes=(2, 2),
        )

        batch = collate_generation(
            examples,
            num_levels=2,
            sid_logq_table=sid_logq,
            user_profiles=profiles,
        )

        self.assertEqual(len(batch["sid_log_q"]), 2)
        self.assertEqual(batch["user_features"].shape, (2, 2))


class ProfileConditionedGeneratorTest(unittest.TestCase):
    def test_legacy_model_has_no_profile_parameters_and_loads_strictly(self):
        first = SemanticTransformerModel(3, model_config(use_user_profile=False))
        second = SemanticTransformerModel(3, model_config(use_user_profile=False))

        second.load_state_dict(first.state_dict(), strict=True)

        self.assertFalse(
            any("user_profile" in name for name in first.state_dict())
        )

    def test_profile_changes_encoded_memory_and_reaches_beam_search(self):
        model = SemanticTransformerModel(3, model_config(use_user_profile=True))
        model.eval()
        history = torch.tensor([[1, 2], [1, 2]])
        positions = torch.tensor([[0, 1], [0, 1]])
        mask = torch.ones(2, 2, dtype=torch.bool)
        profiles = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        memory, _ = model.encode_history(
            history,
            positions,
            mask,
            user_features=profiles,
        )
        self.assertFalse(torch.allclose(memory[0], memory[1]))

        with patch.object(
            model,
            "encode_history",
            wraps=model.encode_history,
        ) as encode_history:
            beams, _ = semantic_beam_search(
                model,
                history,
                positions,
                mask,
                beam_size=2,
                user_features=profiles,
            )

        self.assertEqual(beams.shape, (2, 2, 2))
        self.assertIs(
            encode_history.call_args.kwargs["user_features"],
            profiles,
        )

    def test_profile_model_rejects_missing_invalid_or_nonfinite_features(self):
        model = SemanticTransformerModel(3, model_config(use_user_profile=True))
        history = torch.tensor([[1, 2]])
        positions = torch.tensor([[0, 1]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "required"):
            model.encode_history(history, positions, mask)
        with self.assertRaisesRegex(ValueError, "expected"):
            model.encode_history(
                history,
                positions,
                mask,
                user_features=torch.ones(1, 3),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            model.encode_history(
                history,
                positions,
                mask,
                user_features=torch.tensor([[float("inf"), 0.0]]),
            )

    def test_content_embeddings_follow_generation_item_indices(self):
        embeddings = _initial_item_embeddings(
            pl.DataFrame(
                {
                    "item_id": [20, 10],
                    "normalized_embed": [
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                }
            ),
            {10: 1, 20: 2},
            column="normalized_embed",
            d_model=4,
        )

        self.assertTrue(torch.equal(embeddings[0], torch.zeros(4)))
        self.assertTrue(
            torch.equal(embeddings[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        )
        self.assertTrue(
            torch.equal(embeddings[2], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        )

    def test_content_embeddings_can_be_zero_padded_for_a_larger_model(self):
        embeddings = _initial_item_embeddings(
            pl.DataFrame(
                {
                    "item_id": [10],
                    "normalized_embed": [[1.0, 2.0]],
                }
            ),
            {10: 1},
            column="normalized_embed",
            d_model=4,
            scale=2.0,
            pad_to_model_dim=True,
        )

        self.assertTrue(
            torch.equal(embeddings[1], torch.tensor([2.0, 4.0, 0.0, 0.0]))
        )

    def test_content_padding_requires_a_larger_model_dimension(self):
        with self.assertRaisesRegex(ValueError, "dimension 1"):
            _initial_item_embeddings(
                pl.DataFrame(
                    {
                        "item_id": [10],
                        "normalized_embed": [[1.0, 2.0]],
                    }
                ),
                {10: 1},
                column="normalized_embed",
                d_model=1,
                pad_to_model_dim=True,
            )

    def test_auxiliary_loss_does_not_treat_duplicate_targets_as_negatives(self):
        model = SemanticTransformerModel(3, model_config(use_user_profile=False))
        pooled_history = torch.randn(3, 4)

        loss = model._ranking_loss(
            pooled_history,
            torch.tensor([2, 2, 2]),
        )

        self.assertAlmostEqual(float(loss.loss.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
