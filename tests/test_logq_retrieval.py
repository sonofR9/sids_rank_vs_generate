import math
import os
import platform
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from sids_rank_vs_generate.data.generation import attach_item_log_q
from sids_rank_vs_generate.losses.logq_retrieval import (
    InBatchRetrievalLoss,
    apply_logq_correction,
    concatenate_gathered_candidates,
    corrected_logq_loss,
    probabilities_to_log_q,
    sanitize_log_q,
)


def _distributed_logq_worker(rank: int, init_file: str, result_dir: str) -> None:
    if platform.system() == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        local_size = 2 if rank == 0 else 1
        queries = torch.tensor(
            [[1.0 + rank, 0.5], [0.2, 1.0]][:local_size],
            requires_grad=True,
        )
        items = torch.tensor(
            [[1.0, 0.1], [0.1, 1.0]][:local_size],
            requires_grad=True,
        ) + rank
        log_q = torch.tensor([-1.0, -1.5][:local_size]) - rank
        output = InBatchRetrievalLoss(logq_mode="corrected")(
            queries, items, log_q
        )
        output.loss.backward()
        torch.save(
            {
                "labels": output.labels,
                "columns": output.corrected_logits.size(1),
                "loss_finite": bool(torch.isfinite(output.loss)),
                "query_grad_finite": bool(torch.isfinite(queries.grad).all()),
            },
            Path(result_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


class LogQCorrectionTest(unittest.TestCase):
    def test_none_reproduces_previous_cross_entropy(self) -> None:
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        items = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        output = InBatchRetrievalLoss(
            temperature=0.2,
            logq_mode="none",
            gather_distributed=False,
        )(queries, items)
        expected_logits = F.normalize(queries) @ F.normalize(items).T / 0.2
        expected = F.cross_entropy(expected_logits, torch.arange(2))
        torch.testing.assert_close(output.raw_logits, expected_logits)
        torch.testing.assert_close(output.loss, expected)

    def test_correction_broadcasts_over_document_columns(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        log_q = torch.log(torch.tensor([0.5, 0.1]))
        labels = torch.tensor([0, 1])
        corrected = apply_logq_correction(
            logits, log_q, labels, preserve_positive=False
        )
        torch.testing.assert_close(corrected, logits - log_q[None, :])

    def test_corrected_mode_preserves_only_positive_columns_per_row(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        log_q = torch.log(torch.tensor([0.5, 0.1]))
        labels = torch.tensor([0, 1])
        corrected = apply_logq_correction(
            logits, log_q, labels, preserve_positive=True
        )
        self.assertEqual(corrected[0, 0], logits[0, 0])
        self.assertEqual(corrected[1, 1], logits[1, 1])
        self.assertEqual(corrected[0, 1], logits[0, 1] - log_q[1])
        self.assertEqual(corrected[1, 0], logits[1, 0] - log_q[0])

    def test_uniform_q_adds_constant_only_to_negatives_in_corrected_mode(self) -> None:
        logits = torch.arange(9, dtype=torch.float32).view(3, 3)
        log_q = torch.full((3,), math.log(0.25))
        labels = torch.arange(3)
        corrected = apply_logq_correction(
            logits, log_q, labels, preserve_positive=True
        )
        delta = corrected - logits
        self.assertTrue(torch.equal(delta.diag(), torch.zeros(3)))
        mask = ~torch.eye(3, dtype=torch.bool)
        torch.testing.assert_close(delta[mask], torch.full((6,), -log_q[0]))

    def test_rare_item_receives_larger_correction(self) -> None:
        logits = torch.zeros(2, 2)
        log_q = torch.log(torch.tensor([0.5, 0.001]))
        corrected = apply_logq_correction(
            logits, log_q, torch.arange(2), preserve_positive=False
        )
        self.assertGreater(corrected[0, 1], corrected[0, 0])

    def test_tiny_and_nonfinite_probabilities_are_safe(self) -> None:
        log_q = probabilities_to_log_q(
            torch.tensor([0.0, 1e-40, float("nan"), float("inf")]), eps=1e-8
        )
        safe, fraction = sanitize_log_q(
            torch.tensor([-float("inf"), float("nan"), 2.0, -100.0]),
            eps=1e-8,
        )
        self.assertTrue(torch.isfinite(log_q).all())
        self.assertTrue(torch.isfinite(safe).all())
        self.assertEqual(float(fraction), 1.0)

    def test_batch_size_one_is_finite_in_all_modes(self) -> None:
        query = torch.tensor([[1.0, 2.0]], requires_grad=True)
        item = torch.tensor([[2.0, 1.0]], requires_grad=True)
        for mode in ("none", "standard", "corrected"):
            output = InBatchRetrievalLoss(
                logq_mode=mode, gather_distributed=False
            )(query, item, None if mode == "none" else torch.tensor([0.0]))
            self.assertTrue(torch.isfinite(output.loss))

    def test_gradients_are_finite(self) -> None:
        queries = torch.randn(4, 5, requires_grad=True)
        items = torch.randn(4, 5, requires_grad=True)
        log_q = torch.log(torch.tensor([0.7, 0.2, 0.09, 0.01]))
        output = InBatchRetrievalLoss(
            logq_mode="corrected", gather_distributed=False
        )(queries, items, log_q)
        output.loss.backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(items.grad).all())

    def test_corrected_weight_is_stop_gradient(self) -> None:
        raw = torch.tensor([[0.3, -0.1, 0.7]], requires_grad=True)
        labels = torch.tensor([0])
        log_q = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        corrected = apply_logq_correction(
            raw, log_q, labels, preserve_positive=True
        )
        loss, probability = corrected_logq_loss(raw, corrected, labels)
        actual_gradient = torch.autograd.grad(loss, raw, retain_graph=True)[0]

        negative_logsumexp = torch.logsumexp(corrected[:, 1:], dim=1)
        base_loss = (negative_logsumexp - raw[:, 0]).mean()
        base_gradient = torch.autograd.grad(base_loss, raw)[0]
        expected_gradient = (1.0 - probability).view(-1, 1) * base_gradient
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertFalse(probability.requires_grad)

    def test_bfloat16_encoder_outputs_use_float32_loss_path(self) -> None:
        queries = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
        items = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
        output = InBatchRetrievalLoss(
            logq_mode="standard", gather_distributed=False
        )(queries, items, torch.log(torch.tensor([0.8, 0.19, 0.01])))
        self.assertEqual(output.raw_logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(torch.isfinite(queries.grad).all())


class LogQDataAndDistributedOrderingTest(unittest.TestCase):
    def test_frequency_log_q_is_attached_after_sampling(self) -> None:
        examples = [{"target": 1}, {"target": 1}, {"target": 2}]
        attached = attach_item_log_q(examples)
        self.assertAlmostEqual(attached[0]["log_q"], math.log(2 / 3), places=6)
        self.assertAlmostEqual(attached[2]["log_q"], math.log(1 / 3), places=6)

    def test_rank_order_keeps_items_logq_and_positive_labels_aligned(self) -> None:
        gathered_items = (
            torch.tensor([[10.0], [11.0], [-1.0]]),
            torch.tensor([[20.0], [21.0], [22.0]]),
        )
        gathered_log_q = (
            torch.tensor([-1.0, -1.1, 0.0]),
            torch.tensor([-2.0, -2.1, -2.2]),
        )
        items, log_q, labels = concatenate_gathered_candidates(
            gathered_items,
            gathered_log_q,
            [2, 3],
            rank=1,
            local_batch_size=3,
        )
        torch.testing.assert_close(items[:, 0], torch.tensor([10., 11., 20., 21., 22.]))
        torch.testing.assert_close(log_q, torch.tensor([-1., -1.1, -2., -2.1, -2.2]))
        self.assertTrue(torch.equal(labels, torch.tensor([2, 3, 4])))

    @unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
    def test_two_rank_gather_and_backward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "distributed_init")
            mp.start_processes(
                _distributed_logq_worker,
                args=(init_file, directory),
                nprocs=2,
                join=True,
                start_method="spawn",
            )
            rank_0 = torch.load(Path(directory) / "rank_0.pt", weights_only=True)
            rank_1 = torch.load(Path(directory) / "rank_1.pt", weights_only=True)
        self.assertTrue(torch.equal(rank_0["labels"], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(rank_1["labels"], torch.tensor([2])))
        self.assertEqual(rank_0["columns"], 3)
        self.assertEqual(rank_1["columns"], 3)
        self.assertTrue(rank_0["loss_finite"] and rank_1["loss_finite"])
        self.assertTrue(rank_0["query_grad_finite"] and rank_1["query_grad_finite"])


if __name__ == "__main__":
    unittest.main()
