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


def positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


NUM_LEVELS = positive_int("NUM_LEVELS", 3)
CODEBOOK_SIZE = positive_int("CODEBOOK_SIZE", 256)
NUM_MERGES = positive_int("NUM_MERGES", 512, allow_zero=True)

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
