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

The common preparation filters interactions to items with embeddings, creates
chronological fit/validation/test boundaries, and creates one dense item
mapping. Comparison configs use `catalog_from_fit=True`, so the item core and
catalog are derived from fit only. The legacy pre-test `train` pool remains
available to the standalone SASRec command. Ranking and generation receive the
same leakage-safe catalog and final-test boundary.

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

### MLflow tracking

Enable MLflow for the complete experiment with `--mlflow`. The runner creates
one parent run and child runs for global SID training, DCNv2, TransAct, and
semantic generation. User-aware experiments add child runs for personalized
SID construction and the SID mapper. Custom PyTorch loops log train and
validation metrics once per epoch; the parent run stores final SID and
downstream metrics. Checkpoints and generated files remain in the local report
directory printed by the run.

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

To run a tracking server on the same VM as the experiment:

```bash
mkdir -p "$PWD/mlflow/artifacts"
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "sqlite:///$PWD/mlflow/mlflow.db" \
  --default-artifact-root "$PWD/mlflow/artifacts"
```

Check it and point the experiment at the HTTP endpoint:

```bash
curl --fail http://127.0.0.1:5000/health
python run.py experiments/user_aware_transformer_mapper_full.py \
  --mlflow \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --mlflow-experiment-name sid-comparison
```

For runs on several VMs, use one tracking server reachable over a private
network or SSH tunnel. Do not expose an unauthenticated MLflow port to the
public internet. Every worker receives the same `--mlflow-tracking-uri`.

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

Generation predicts the target item's SID autoregressively from item history.
Collision resolution can append a final disambiguation level; without it, beams
are expanded through the SID-to-items mapping for item recall.

Artifacts use this layout:

```text
reports/experiments/50m/RQKMeansPipeline/<UTC-run-id>/
  metrics.json
  semantic_ids.parquet
  user_semantic_ids_fit.parquet        # user-aware only
  ranking/
    dcnv2.pt
    transact.pt
  generation/
    best_model.pt
    best_mapper.pt                     # Transformer mapper only
    semantic_ids.parquet
    user_semantic_ids_observed.parquet # user-aware only
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

## User-aware Semantic IDs

The optional user-aware pipeline is fitted from interactions strictly before
the validation boundary. The global item SID remains the ordinary
`ISidPipeline`; personalized IDs are an additional pair-level representation:

```python
from sids_rank_vs_generate.experiment import ExperimentConfig
from sids_rank_vs_generate.sids import (
    RQKMeansPipeline,
    UserAwareRQKMeansPipeline,
)

EXPERIMENT = ExperimentConfig(
    catalog_from_fit=True,
    sid_pipeline=RQKMeansPipeline(...),
    user_sid_pipeline=UserAwareRQKMeansPipeline(...),
)
```

Ranking uses personalized candidate `usid_*` features while preserving the
global-SID history representation. Generation predicts `SID_u`, applies a
train-only `(user, SID_u) -> SID` mapper, and resolves the global SID to catalog
items. The frequency mapper remains the default treatment:

```bash
python run.py experiments/user_aware_rqkmeans_full.py
```

A learned conditional Transformer mapper is available as a separate,
directly comparable experiment. It conditions on the frozen train-only user
profile and `SID_u`, then uses catalog-constrained decoding so every returned
global SID exists:

```bash
python run.py experiments/user_aware_transformer_mapper_full.py
```

Content RQ-KMeans, content RQ-VAE, and the no-user pair control are configured
in `experiments/rqkmeans_full.py`, `experiments/rqvae_full.py`, and
`experiments/user_aware_no_user_control.py` respectively. Mapper selection is
part of `GenerationConfig` through `UserSidMapperConfig(kind=...)`.

MLflow is optional and never changes training behavior. Install
`pip install -e '.[mlflow]'` and set
`tracking=MLflowConfig(enabled=True, tracking_uri=...)` on `ExperimentConfig`,
or use the command-line flags shown above.

## SASRec baseline

The earlier non-SID SASRec experiment remains available separately:

```bash
sasrec-baseline --epochs 10
```

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -t . -v
```

The suite includes end-to-end CPU runs for content-only and user-aware
RQ-KMeans. They evaluate both ranking models, generation, Transformer mapping,
and the complete tracking run hierarchy.
