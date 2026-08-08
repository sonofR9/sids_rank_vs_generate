import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sids_rank_vs_generate.evaluation.beam_search import (
    SemanticIDCatalog,
    semantic_beam_search,
)
from sids_rank_vs_generate.evaluation.personalized_generation import (
    _rank_personalized_items,
)
from sids_rank_vs_generate.evaluation.semantic_validation_with_collision import (
    validate_semantic_model_with_collisions,
)
from sids_rank_vs_generate.models.semantic_transformer_model import (
    SemanticTransformerModel,
)


class _DummyEncoder(nn.Module):
    def forward(self, history, positions, mask):
        memory = torch.zeros(
            history.size(0),
            history.size(1),
            2,
            device=history.device,
        )
        return memory, memory[:, -1]


class _DummyDecoder(nn.Module):
    def forward(self, tgt, **kwargs):
        return tgt


class _FixedHead(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values.unsqueeze(0).expand(inputs.size(0), -1)


class _DummyBeamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 2
        self.codebook_sizes = (2, 2)
        self.bos_id = 4
        self.register_buffer("codebook_offsets", torch.tensor([0, 2]))
        self.encoder = _DummyEncoder()
        self.decoder_embed = nn.Embedding(5, 2)
        self.positions_embed = nn.Embedding(2, 2)
        self.decoder = _DummyDecoder()
        self.heads = nn.ModuleList([
            _FixedHead([2.0, 1.0]),
            _FixedHead([2.0, 1.0]),
        ])

    @staticmethod
    def causal_mask(length, device):
        return torch.zeros(length, length, device=device)


class SemanticIDCatalogTest(unittest.TestCase):
    def test_constrained_search_never_returns_an_invalid_path(self):
        model = _DummyBeamModel()
        history = torch.ones(1, 1, dtype=torch.long)
        positions = torch.zeros_like(history)
        mask = torch.ones_like(history, dtype=torch.bool)
        catalog = SemanticIDCatalog(
            semantic_ids=[(0, 1), (1, 0)],
            codebook_sizes=(2, 2),
        )

        unconstrained, _ = semantic_beam_search(
            model,
            history,
            positions,
            mask,
            beam_size=1,
        )
        constrained, _ = semantic_beam_search(
            model,
            history,
            positions,
            mask,
            beam_size=1,
            sid_catalog=catalog,
        )

        self.assertEqual(unconstrained[0, 0].tolist(), [0, 0])
        self.assertEqual(constrained[0, 0].tolist(), [0, 1])

    def test_catalog_rejects_out_of_range_tokens(self):
        with self.assertRaises(ValueError):
            SemanticIDCatalog([(0, 2)], codebook_sizes=(2, 2))


class SemanticTransformerRecallFeaturesTest(unittest.TestCase):
    def test_content_semantic_history_and_auxiliary_loss_work_together(self):
        cfg = SimpleNamespace(
            num_levels=2,
            rq_num_levels=2,
            codebook_sizes=(2, 2),
            codebook_size=2,
            d_model=4,
            d_ff=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            max_history_len=3,
            semantic_history_weight=1.0,
            auxiliary_item_loss_weight=0.2,
            ranking_temperature=0.07,
        )
        initial_embeddings = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        item_sid_codes = torch.tensor([
            [-1, -1],
            [0, 0],
            [0, 1],
            [1, 0],
        ])
        model = SemanticTransformerModel(
            num_items=3,
            cfg=cfg,
            initial_item_embeddings=initial_embeddings,
            item_sid_codes=item_sid_codes,
        )

        self.assertTrue(torch.equal(
            model.encoder.item_emb.weight,
            initial_embeddings,
        ))

        output = model(
            history=torch.tensor([[1, 2], [2, 3]]),
            positions=torch.tensor([[0, 1], [0, 1]]),
            mask=torch.ones(2, 2, dtype=torch.bool),
            sid=torch.tensor([[0, 1], [1, 0]]),
            target=torch.tensor([2, 3]),
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIsNotNone(output["ranking_loss"])
        self.assertEqual(output["preds"].shape, (2, 2))


class _PerfectSemanticModel(nn.Module):
    def forward(self, history, positions, mask, sid, target=None):
        return {"preds": sid}


class _RecordingMapper:
    def __init__(self) -> None:
        self.requests = []

    def predict_topk_batch(self, requests, k):
        self.requests.extend(requests)
        return [[(int(user_sid[0]),)] for _uid, user_sid in requests]


class SemanticValidationMetricsTest(unittest.TestCase):
    def test_personalized_mapping_stops_after_same_top_k(self):
        beams = torch.tensor([
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
        ])
        sid_to_items = {
            (0,): [10],
            (1,): [11],
            (2,): [12],
            (3,): [13],
        }
        chunked_mapper = _RecordingMapper()
        full_mapper = _RecordingMapper()

        chunked = _rank_personalized_items(
            uids=[1, 2],
            beams=beams,
            mapper=chunked_mapper,
            sid_to_items=sid_to_items,
            recall_k=2,
            mapper_top_k=1,
            mapper_beam_chunk_size=2,
        )
        full = _rank_personalized_items(
            uids=[1, 2],
            beams=beams,
            mapper=full_mapper,
            sid_to_items=sid_to_items,
            recall_k=2,
            mapper_top_k=1,
            mapper_beam_chunk_size=4,
        )

        self.assertEqual(chunked, full)
        self.assertEqual(chunked, [[10, 11], [10, 11]])
        self.assertEqual(len(chunked_mapper.requests), 4)
        self.assertEqual(len(full_mapper.requests), 8)

    def test_user_metrics_are_not_divided_by_prediction_count(self):
        batch = {
            "history": torch.tensor([[1], [2]]),
            "positions": torch.zeros(2, 1, dtype=torch.long),
            "mask": torch.ones(2, 1, dtype=torch.bool),
            "sid": torch.tensor([[0, 0], [0, 0]]),
            "target": torch.tensor([1, 2]),
        }
        beams = torch.tensor([
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ])
        cfg = SimpleNamespace(
            device="cpu",
            use_bf16=False,
            num_beams=2,
            recall_k=2,
        )
        catalog = SemanticIDCatalog(
            semantic_ids=[(0, 0), (1, 1)],
            codebook_sizes=(2, 2),
        )

        with patch(
            "sids_rank_vs_generate.evaluation."
            "semantic_validation_with_collision.semantic_beam_search",
            return_value=(beams, torch.zeros(2, 2)),
        ) as beam_search:
            metrics = validate_semantic_model_with_collisions(
                model=_PerfectSemanticModel(),
                val_loader=[batch],
                cfg=cfg,
                sid_to_items={(0, 0): [1, 2]},
                sid_catalog=catalog,
            )

        self.assertIs(beam_search.call_args.kwargs["sid_catalog"], catalog)
        self.assertEqual(metrics["top1_sid_exact"], 1.0)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["Effective vocab size"], 2)


if __name__ == "__main__":
    unittest.main()
