from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.evaluation.mapper_inference_diagnostic import (
    _ordered_unique_items,
    diagnose_mapper_inference,
    mapper_inference_mlflow_summary,
)
from sids_rank_vs_generate.evaluation.personalized_generation import (
    _rank_personalized_items,
)
from sids_rank_vs_generate.sids.transformer_user_sid_mapper import (
    UserSidMapperConfig,
)


class _FakeMapper:
    def __init__(self) -> None:
        self._predictions = {
            (1, (0,)): [(10,), (91,)],
            (1, (8,)): [(92,)],
            (2, (1,)): [(42,), (20,)],
            (2, (8,)): [(40,), (41,)],
            (3, (2,)): [(30,), (93,)],
            (3, (7,)): [(30,), (94,)],
            (3, (6,)): [(95,)],
        }
        self._seen = {(1, (0,)), (2, (1,))}
        self.requested_k = []

    def predict_topk_batch(self, requests, k):
        self.requested_k.append(k)
        return [
            self._predictions[(uid, tuple(int(code) for code in sid))][:k]
            for uid, sid in requests
        ]

    def is_seen(self, uid, user_sid):
        return (uid, tuple(int(code) for code in user_sid)) in self._seen


class _Model(nn.Module):
    pass


class _ExactMapper:
    def predict_topk_batch(self, requests, k):
        return [
            [
                (1000 + int(sid[0]),),
                (2000 + int(sid[0]),),
                (3000 + int(sid[0]),),
            ][:k]
            for _uid, sid in requests
        ]

    def is_seen(self, uid, user_sid):
        return int(uid) == 1


class MapperInferenceDiagnosticTest(unittest.TestCase):
    def test_measures_gates_coverage_and_real_truncation(self):
        batch = {
            "uid": [1, 2, 3],
            "history": torch.ones(3, 1, dtype=torch.long),
            "positions": torch.zeros(3, 1, dtype=torch.long),
            "mask": torch.ones(3, 1, dtype=torch.bool),
            "sid": torch.tensor([[0], [1], [2]]),
            "global_sid": torch.tensor([[10], [20], [30]]),
            "target": torch.tensor([100, 200, 300]),
        }
        beams = torch.tensor(
            [
                [[0], [8]],
                [[8], [1]],
                [[7], [6]],
            ]
        )
        cfg = SimpleNamespace(device="cpu", use_bf16=False)
        sid_to_items = {
            (10,): [100],
            (20,): [200],
            (30,): [300],
            (40,): [400],
            (41,): [401],
            (42,): [402],
            (91,): [901],
            (92,): [902],
            (93,): [903],
            (94,): [904],
            (95,): [905],
        }

        mapper = _FakeMapper()
        with patch(
            "sids_rank_vs_generate.evaluation.mapper_inference_diagnostic."
            "semantic_beam_search",
            return_value=(beams, torch.zeros(3, 2)),
        ) as beam_search:
            result = diagnose_mapper_inference(
                _Model(),
                [batch],
                cfg,
                mapper,
                sid_to_items,
                beam_cutoffs=(1, 2),
                mapper_cutoffs=(1, 2),
                recommendation_cutoffs=(1, 3),
            )

        self.assertEqual(beam_search.call_args.kwargs["beam_size"], 2)
        self.assertEqual(mapper.requested_k, [1, 1, 2, 2])
        self.assertEqual(result["examples"], {"all": 3, "seen": 2, "unseen": 1})
        self.assertEqual(result["E_B"]["1"]["all"]["rate"], 1 / 3)
        self.assertEqual(result["E_B"]["2"]["all"]["rate"], 2 / 3)
        self.assertEqual(
            result["teacher_forced_mapper_top_k"]["1"]["all"]["rate"],
            2 / 3,
        )
        self.assertEqual(
            result["teacher_forced_mapper_top_k"]["1"]["seen"]["rate"],
            0.5,
        )
        self.assertEqual(
            result["teacher_forced_mapper_top_k"]["2"]["all"]["rate"],
            1.0,
        )

        b1k1 = result["grid"]["1"]["1"]
        self.assertEqual(
            b1k1["item_candidate_coverage_before_truncation"]["all"]["rate"],
            2 / 3,
        )
        self.assertEqual(b1k1["final_recall"]["1"]["all"]["rate"], 2 / 3)

        b2k2 = result["grid"]["2"]["2"]
        self.assertEqual(
            b2k2["item_candidate_coverage_before_truncation"]["all"]["rate"],
            1.0,
        )
        self.assertEqual(b2k2["final_recall"]["3"]["all"]["rate"], 2 / 3)
        self.assertEqual(
            b2k2["conditional_on_E_B"]["item_candidate_coverage_before_truncation"][
                "all"
            ]["rate"],
            1.0,
        )
        self.assertEqual(
            b2k2["conditional_on_E_B"]["final_recall"]["3"]["all"]["rate"],
            0.5,
        )
        self.assertEqual(b2k2["ordering_loss"]["3"]["all"]["rate"], 1 / 3)
        self.assertEqual(
            b2k2["conditional_on_E_B"]["ordering_loss"]["3"]["all"]["rate"],
            0.5,
        )
        self.assertEqual(
            b2k2["conditional_on_not_E_B"]["final_recall"]["3"]["all"]["rate"],
            1.0,
        )
        self.assertEqual(
            b2k2["conditional_on_not_E_B"]["final_recall"]["3"]["all"]["examples"],
            1,
        )

    def test_b150_k3_n100_matches_standard_personalized_recall(self):
        beam_values = list(range(150))
        beams = torch.tensor([[beam_values], [beam_values]]).transpose(1, 2)
        targets = [1120, 1000]
        batch = {
            "uid": [1, 2],
            "history": torch.ones(2, 1, dtype=torch.long),
            "positions": torch.zeros(2, 1, dtype=torch.long),
            "mask": torch.ones(2, 1, dtype=torch.bool),
            "sid": torch.tensor([[120], [0]]),
            "global_sid": torch.tensor([[1120], [1000]]),
            "target": torch.tensor(targets),
        }
        sid_to_items = {
            (offset + code,): [offset + code]
            for offset in (1000, 2000, 3000)
            for code in range(150)
        }
        mapper = _ExactMapper()

        standard_ranked = _rank_personalized_items(
            uids=batch["uid"],
            beams=beams,
            mapper=mapper,
            sid_to_items=sid_to_items,
            recall_k=100,
            mapper_top_k=3,
            mapper_beam_chunk_size=16,
        )
        standard_recall = sum(
            target in ranked
            for target, ranked in zip(targets, standard_ranked, strict=True)
        ) / len(targets)

        with patch(
            "sids_rank_vs_generate.evaluation.mapper_inference_diagnostic."
            "semantic_beam_search",
            return_value=(beams, torch.zeros(2, 150)),
        ):
            diagnostic = diagnose_mapper_inference(
                _Model(),
                [batch],
                SimpleNamespace(device="cpu", use_bf16=False),
                mapper,
                sid_to_items,
                beam_cutoffs=(150,),
                mapper_cutoffs=(3,),
                recommendation_cutoffs=(100,),
            )

        diagnostic_recall = diagnostic["grid"]["150"]["3"]["final_recall"]["100"][
            "all"
        ]["rate"]
        self.assertEqual(standard_recall, 0.5)
        self.assertEqual(diagnostic_recall, standard_recall)
        summary = mapper_inference_mlflow_summary(diagnostic)
        self.assertEqual(summary["E_B"]["150"], 1.0)
        self.assertEqual(summary["grid"]["150"]["3"]["final_recall_at_100"], 0.5)
        self.assertEqual(
            summary["grid"]["150"]["3"]["ordering_loss_at_100"],
            0.5,
        )

    def test_ordering_deduplicates_items_without_reordering(self):
        ranked, global_candidates = _ordered_unique_items(
            [
                [(1,), (2,)],
                [(1,), (3,)],
            ],
            {
                (1,): [10, 11],
                (2,): [11, 12],
                (3,): [13],
            },
            mapper_top_k=2,
        )

        self.assertEqual(ranked, [10, 11, 12, 13])
        self.assertEqual(global_candidates, {(1,), (2,), (3,)})

    def test_requires_both_saved_checkpoints(self):
        with self.assertRaisesRegex(ValueError, "saved generator"):
            GenerationConfig(run_mapper_inference_diagnostic=True)

        config = GenerationConfig(
            run_mapper_inference_diagnostic=True,
            load_checkpoint_path=Path("generator.pt"),
            mapper=UserSidMapperConfig(
                kind="transformer",
                load_checkpoint_path=Path("mapper.pt"),
            ),
        )

        self.assertTrue(config.run_mapper_inference_diagnostic)


if __name__ == "__main__":
    unittest.main()
