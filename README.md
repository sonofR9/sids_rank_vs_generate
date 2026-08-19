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

By default, preprocessing and history construction use the latest 365 days. A
regular run trains on the 270 days immediately preceding the fixed 7-day test and
does not create a validation split. Hyperparameter-tuning configurations set
`RankingConfig(hyperparameter_tuning=True)` to train on 270 days and evaluate the
following fixed 7-day validation window without inspecting test. Earlier
interactions remain available when constructing each candidate's history. The
ranking defaults use model-specific FP32 schedules selected on that 7-day
validation window: DCNv2 uses batch 32,768 for two epochs at `2e-3`, while
TransAct uses batch 8,192 for one epoch at `5e-4`.

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

### Collaborative-aware RQ-KMeans (OneRec)

`CollaborativeAwareRQKMeansPipeline` mines recent co-interaction pairs only
from the temporal train split, aligns content embeddings with a symmetric
in-batch contrastive objective, and sends the aligned vectors to RQ-KMeans.
This follows the collaborative-aware representation stage in
[OneRec, Section 2.1.1](https://arxiv.org/abs/2506.13695).
Pair-mining and alignment parameters are declared in
`CollaborativeAlignmentConfig`; the shared runner records the temporal cutoff,
pair statistics, losses, and before/after cosine diagnostics.

Run the aligned experiment and the content-only control with the same fixed
ranking and generation stages:

```bash
python run.py experiments/collaborative_rqkmeans_full.py
python run.py experiments/rqkmeans_full.py
```

The resulting `metrics.json` files contain directly comparable `ranking` and
`generation` sections. Use `--sid-only` for pair/alignment/RQ-KMeans smoke
checks without downstream training.

For the poster-compatible 33,148-item comparison, use two GPUs to run the
aligned method and its content-only control concurrently. Each process uses one
GPU and writes a complete parent run with nested SID, DCNv2, TransAct, and
generation runs to MLflow:

```bash
tmux new -s onerec-pairs
PYTHON_BIN=.venv/bin/python \
ALIGNED_GPU=0 BASELINE_GPU=1 \
MLFLOW_TRACKING_URI=http://51.250.47.231:5000 \
bash scripts/run_collaborative_rqkmeans_pair.sh \
  2>&1 | tee onerec-pair-comparison.log
```

`RUN_SET=aligned` or `RUN_SET=baseline` runs only one side. The launcher first
prepares the shared Yambda cache, avoiding concurrent dataset downloads.

After both parent runs finish, export the new MLflow point and rebuild the
combined figures:

```bash
python scripts/analyze_generation_sid_quality.py \
  --tracking-uri http://51.250.47.231:5000 \
  --generation-experiments sid-comparison \
  --output-dir reports/figures/generation/one_rec_pair_alignment

python scripts/analyze_ranking_sid_quality.py \
  --tracking-uri http://51.250.47.231:5000 \
  --quality-experiments sid-comparison \
  --ranking-experiments sid-comparison \
  --output-dir reports/figures/ranking/one_rec_pair_alignment

python scripts/plot_all_generation_methods.py \
  --raw-report reports/figures/generation/one_rec_pair_alignment/matched_generation_runs.csv \
  --ranking-report reports/figures/ranking/one_rec_pair_alignment/matched_runs.csv \
  --transact-report reports/figures/ranking/one_rec_pair_alignment/matched_runs.csv
```

### Metrics for precomputed SIDs

To measure SID assignments that have already been generated, place their
Parquet files in one directory. Each file must contain one unique `item_id` row
and integer `sid_0`, `sid_1`, ... columns. Run the metrics-only mode with the
catalog embedding Parquet used to create those IDs:

```bash
python run.py \
  --sids-metrics-dir path/to/precomputed_sids \
  --embeddings-path path/to/embeddings.parquet
```

This skips SID training and downstream evaluation. It calculates all common
collision, uniqueness, codebook-quality, path-similarity, and
`intra_code_similarity_v2` metrics for every Parquet and writes the combined
report to `path/to/precomputed_sids/sid_metrics.json`.

The output includes the existing report metrics (`ICR`, collision rate and
sizes, CUR, path Gini/entropy, prefix coherence, and validity) plus these new
diagnostics:

- `collision_item_rate`: fraction of catalog items that belong to a collided
  complete SID path; lower is better.
- `pair_collision_probability`: probability that two distinct random items
  receive the same complete path; lower is better and heavily penalizes large
  collision groups.
- `pair_refinement_rate_at_<level>_layer`: fraction of item pairs sharing the
  preceding prefix that the current level separates; higher is better.
- `mean_active_branch_fraction_at_<level>_layer`: mean fraction of available
  child codes used under each occupied parent prefix; higher indicates broader
  local branching.
- `normalized_conditional_entropy_at_<level>_layer`: item-weighted balance of
  the new branches after the preceding prefix is known; higher indicates less
  branch collapse.
- `adjusted_semantic_variance_explained_at_<level>`: degrees-of-freedom-adjusted
  clustering R-squared of content embeddings under prefixes of that depth. It
  rewards compact, separated semantic groups without treating all-singleton
  prefixes as perfect; higher is better and singleton-only depths are `null`.

Use `--recursive-sids` for nested folders, `--metrics-output` to choose another
JSON path, or explicit codebook sizes when observed ranges do not describe the
full codebooks:

```bash
python run.py \
  --sids-metrics-dir path/to/precomputed_sids \
  --embeddings-path path/to/embeddings.parquet \
  --recursive-sids \
  --codebook-sizes 256,256,256 \
  --metrics-output reports/precomputed_sid_metrics.json
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
epoch; the parent run stores final SID and downstream metrics, plus
`sid_mapping/semantic_ids.parquet` and `sid_mapping/metrics.json`. The Parquet
has the original catalog `item_id`, `catalog_item_idx`, and `sid_0`, ... columns.
Checkpoints and other generated files remain in the local report directory printed
by the run.

To export all SID mappings without ranking or generation, use
`scripts/export_sid_mapping_matrix.sh`. It runs the historical SID matrix and
uploads each mapping under `sid_mapping/semantic_ids.parquet`:

```bash
tmux new -s sid-mappings
RUN_SET=all GPU=0 bash scripts/export_sid_mapping_matrix.sh 2>&1 | tee sid-mappings.log
```

The same script accepts `RUN_SET=rqkmeans`, `rqvae`, `r3vae`, or `balance`.

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
Every historical item is expanded into one transformer token per SID level;
each level has its own learned embedding table.
Ranking defaults are tuned for a 48 GiB V100: DCNv2 uses batch size 32,768 and
TransAct uses batch size 8,192. Both train in FP32 with PyTorch float32 matmul
precision set to `high`.
Pairwise accuracy on adjacent opposite-label events within 15-minute user gaps
is the primary reporting metric. ROC-AUC selects checkpoints during tuning;
PR-AUC and log loss remain secondary diagnostics. Attention is
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
