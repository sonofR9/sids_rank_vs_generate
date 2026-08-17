import os

from sids_rank_vs_generate.experiment import (
    ExperimentConfig,
    YambdaDatasetConfig,
    TrainingConfig,
)

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig

from sids_rank_vs_generate.sids import (
    BPE_RQVAEConfig,
    BPERQVAEPipeline,
)


NUM_LEVELS = int(os.environ.get("NUM_LEVELS", "3"))
CODEBOOK_SIZE = int(os.environ.get("CODEBOOK_SIZE", "256"))
NUM_MERGES = int(os.environ.get("NUM_MERGES", "512"))

COLLISION_RESOLUTION = os.environ.get(
    "COLLISION_RESOLUTION",
    "none",
)

ASSIGNMENT_METHOD = os.environ.get(
    "ASSIGNMENT_METHOD",
    "nearest",
)


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(size="50m"),

    sid_pipeline=BPERQVAEPipeline(
        BPE_RQVAEConfig(
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
            bpe_num_merges=NUM_MERGES,
            bpe_min_frequency=2,
            assignment_method=ASSIGNMENT_METHOD,
        )
    ),

    training=TrainingConfig(
        generation=GenerationConfig(
            resolve_collisions=COLLISION_RESOLUTION != "none",
        )
    ),
)
