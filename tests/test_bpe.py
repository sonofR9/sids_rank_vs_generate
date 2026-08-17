from __future__ import annotations

import unittest

import polars as pl

from sids_rank_vs_generate.sids import BPE_RQVAEConfig, BPERQVAEPipeline
from sids_rank_vs_generate.sids.bpe import BPE


class BPETest(unittest.TestCase):
    def test_fit_transform_accepts_a_generator(self) -> None:
        sequences = (
            [("L0", 1), ("L1", 2)]
            for _ in range(2)
        )

        encoded = BPE(num_merges=1, codebook_size=4).fit_transform(sequences)

        self.assertEqual(encoded, [[("bpe", 0)], [("bpe", 0)]])

    def test_transform_rejects_invalid_base_tokens(self) -> None:
        bpe = BPE(num_merges=1, codebook_size=4).fit(
            [[("L0", 1), ("L1", 2)]]
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            bpe.transform([[("L0", 4)]])

    def test_polars_output_preserves_positions_and_uses_pad(self) -> None:
        frame = pl.DataFrame({"item_id": [10, 11], "sid_0": [1, 1], "sid_1": [2, 2], "sid_2": [3, 3]})
        bpe = BPE(num_merges=1, codebook_size=4).fit_polars(
            frame, ["sid_0", "sid_1", "sid_2"]
        )

        result = bpe.transform_polars(
            frame, ["sid_0", "sid_1", "sid_2"], pad_id=bpe.pad_id
        )

        self.assertEqual(result["sid_0"].to_list(), [4, 4])
        self.assertEqual(result["sid_1"].to_list(), [5, 5])
        self.assertEqual(result["sid_2"].to_list(), [3, 3])

    def test_pipeline_exposes_post_bpe_vocabulary_capacity(self) -> None:
        pipeline = BPERQVAEPipeline(
            BPE_RQVAEConfig(num_levels=10, codebook_size=32, bpe_num_merges=512)
        )

        self.assertEqual(pipeline.sid_codebook_sizes, (545,) * 10)


if __name__ == "__main__":
    unittest.main()
