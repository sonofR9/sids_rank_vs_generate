from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    YambdaDatasetConfig,
)

from sids_rank_vs_generate.sids import (
    BPE_RQVAEConfig,
    BPERQVAEPipeline,
)

EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),
    sid_pipeline=BPERQVAEPipeline(
        BPE_RQVAEConfig(
            num_levels=3,
            codebook_size=256,
            bpe_num_merges=512,
            bpe_min_frequency=2,
        )
    ),
)