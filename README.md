# SIDs: rank vs generate

This repository compares semantic item IDs on the same Yambda catalog in two
downstream tasks:

- ranking with DCNv2 and a history-aware TransAct-like transformer;
- autoregressive candidate generation with a TIGER-like semantic transformer.

Every experiment is defined by a Python file and launched through the single
repository entry point, `run.py`.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

The default dataset is the official `yandex/yambda` `flat/50m/likes.parquet`
file. The runner prepares a compact item-embedding cache from the global
precomputed embeddings supplied as `embeddings.parquet`:

```text
~/.cache/sids_rank_vs_generate/datasets/yambda-50m/
  embeddings.parquet
```

The common preparation filters interactions to items with embeddings, applies
the configured item-core filter, creates one dense item mapping, and creates one
chronological train/test boundary. Ranking and generation receive stage-specific
views of this same prepared data and catalog.

## Experiment configuration

An experiment file contains the instantiated SID pipeline and downstream
configuration. SID statistics are always calculated by the runner.

```python
from pathlib import Path

from sids_rank_vs_generate.evaluation.generation_stage import GenerationConfig
from sids_rank_vs_generate.evaluation.ranking_stage import RankingConfig
from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(
        source_embeddings_path=Path("embeddings.parquet"),
    ),
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(num_levels=3, codebook_size=256),
    ),
    ranking=RankingConfig(models=("dcnv2", "transact")),
    generation=GenerationConfig(),
    seed=42,
)
```

Run the included configuration with:

```bash
python run.py experiments/rqkmeans_full.py
```

Set `ranking=None` or `generation=None` to skip that downstream task. A config
can also point `interactions_path` and `embeddings_path` at explicitly prepared
Parquet files.

## Pipeline

For each run, the runner:

1. prepares Yambda interactions and item embeddings;
2. calls `sid_pipeline.train(item_features)`;
3. calls `sid_pipeline.update(item_features)` and validates its `sid_0`, ...
   integer columns;
4. saves the semantic IDs and automatically computes utilization, uniqueness,
   collision, singleton, and collision-percentile statistics;
5. trains every enabled ranking model;
6. trains and evaluates the generation model when enabled;
7. writes one unified `metrics.json`.

Ranking examples pair each positive next interaction with sampled catalog
negatives. The plain DCNv2 consumes the user and candidate SID levels. The
TransAct-like model uses the same semantic-code embedding for both the candidate
and item history.

Generation predicts the target item's SID autoregressively from item history.
Collision resolution can append a final disambiguation level; without it, beams
are expanded through the SID-to-items mapping for item recall.

Artifacts use this layout:

```text
reports/experiments/50m/RQKMeansPipeline/<UTC-run-id>/
  metrics.json
  semantic_ids.parquet
  ranking/
    dcnv2.pt
    transact.pt
  generation/
    best_model.pt
    semantic_ids.parquet
```

## Other SID pipelines

The package also exposes `RQVAEPipeline` and `PlumRQVAEPipeline`. New methods
implement `ISidPipeline`:

```python
class MySidPipeline(ISidPipeline):
    def train(self, df: pl.DataFrame) -> None:
        ...

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        # item_id plus integer sid_0, ..., sid_n columns
        ...
```

## SASRec baseline

The earlier non-SID SASRec experiment remains available separately:

```bash
sasrec-baseline --epochs 10
```

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -t . -v
```

The suite includes an end-to-end CPU run that creates RQKMeans IDs and evaluates
both ranking models and generation in one experiment.
