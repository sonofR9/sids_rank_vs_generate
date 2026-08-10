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

Install the optional MLflow integration when experiment tracking is needed:

```bash
pip install -e '.[mlflow]'
```

## Data

The generation history comes from the official `yandex/yambda`
`flat/50m/likes.parquet` file. Ranking uses the prepared
`matfu21/yambda-50m-lag-features` DCNv2 listens together with the official
Yambda artist and album mappings. The runner also downloads the global
precomputed embeddings once and prepares a compact item-embedding cache:

```text
~/.cache/sids_rank_vs_generate/datasets/yambda-50m/
  embeddings.parquet
```

The common preparation filters interactions to items with embeddings, applies
the configured item-core filter, creates one dense item mapping, and creates one
chronological train/test boundary. Ranking and generation receive stage-specific
views of this same prepared data and catalog.

By default, preprocessing and history construction use the latest 365 days.
Ranking candidates are read only for the 60 training days preceding the 7-day
test boundary; the earlier interactions remain available when constructing each
candidate's preceding history.

## Experiment configuration

An experiment file contains the instantiated SID pipeline. The two ranking
models and the generation model are fixed across experiments; only training
hyperparameters can optionally be overridden. SID statistics are always
calculated by the runner.

```python
from sids_rank_vs_generate.experiment import ExperimentConfig, YambdaDatasetConfig
from sids_rank_vs_generate.sids import RQKMeansConfig, RQKMeansPipeline


EXPERIMENT = ExperimentConfig(
    dataset=YambdaDatasetConfig(),
    sid_pipeline=RQKMeansPipeline(
        RQKMeansConfig(num_levels=3, codebook_size=256),
    ),
)
```

The common runner always uses the fixed seed `42`; experiments cannot override
it, so SID construction and downstream model initialization are comparable.

Run the included configuration with:

```bash
python run.py experiments/rqkmeans_full.py
```

Both tasks run by default. Runtime flags skip stages or bound both interaction
sources for a faster end-to-end check:

```bash
python run.py experiments/rqkmeans_full.py --max-rows 200000
python run.py experiments/rqkmeans_full.py --skip-generation
python run.py experiments/rqkmeans_full.py --skip-ranking
```

### RQ-VAE purely semantic indexing

RQ-VAE supports three final assignment policies over the same trained encoder
and codebooks: vanilla nearest-centroid IDs, recursive residual searching (RRS),
and exhaustive candidate matching (ECM). RRS and ECM resolve collisions without
appending a non-semantic conflict token.

```bash
python run.py experiments/rqvae_psi.py:NEAREST
python run.py experiments/rqvae_psi.py:RRS
python run.py experiments/rqvae_psi.py:ECM
```

The SID result includes ICR before and after assignment, collision-bucket size,
reassigned-item count, and residual-norm distortion. RRS is the practical
full-catalog default. ECM evaluates `top_k ** num_levels` paths per item and is
best suited to controlled comparisons.

### MLflow tracking

Enable MLflow for the complete experiment with `--mlflow`. The runner creates
one parent run and child runs for SID training, DCNv2, TransAct, and semantic
generation. Custom PyTorch loops log their train and validation metrics once per
epoch; the parent run stores final SID and downstream metrics. Checkpoints and
other generated files remain in the local report directory printed by the run.

```bash
python run.py experiments/rqkmeans_full.py \
  --max-rows 200000 \
  --mlflow
```

Use a shared tracking server and explicit names when needed:

```bash
python run.py experiments/rqkmeans_full.py \
  --mlflow \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-experiment-name sid-comparison \
  --mlflow-run-name rqkmeans-l3-c256
```

MLflow is optional and disabled by default, so tests and ordinary runs do not
import it. Python experiment configs can enable the same behavior with
`MLflowConfig(enabled=True)` in `ExperimentConfig.tracking`.

With no explicit tracking URI, current MLflow versions use a local
`sqlite:///mlflow.db` store. Open the local UI after a run with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

A config can also point at explicitly prepared Parquet files for tests or local
development.

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

Ranking uses the existing listen labels and all prepared DCNv2 lag features.
Both models consume the same dense counters, user and candidate SID features,
and multivalent artist/album IDs. The TransAct variant adds the latest like
history strictly preceding each ranking timestamp to that same DCNv2 input.
Ranking defaults are tuned for a 48 GiB V100: batch size 32,768, learning rate
2e-3, and FP32 training with PyTorch float32 matmul precision set to `high`.
Pairwise accuracy on adjacent opposite-label
events within 15-minute user gaps is the primary checkpoint and reporting
metric; ROC-AUC, PR-AUC, and log loss remain secondary diagnostics. Attention is
explicitly restricted to the
math backend, so runs do not depend on Flash Attention support. Columnar batch
assembly is prefetched on CPU while the current batch trains on GPU.

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
