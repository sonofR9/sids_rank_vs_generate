# SIDS rank vs generate

The repository contains shared interaction preprocessing, reproducible
sequential recommendation experiments, and the completed Transformer homework
notebook.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Datasets, archives, checkpoints, and generated metric JSON files are local
artifacts and must not be committed.

## Shared data

The common runner selects only the Yambda scale. Dataset paths, the 5-core item
filter, the final 7-day split, and the random seed are shared by all SID
experiments.

Prepared files live in one cache:

```text
~/.cache/sids_rank_vs_generate/datasets/
  yambda-50m/
    interactions.parquet
    embeddings.parquet
  yambda-500m/
    interactions.parquet
    embeddings.parquet
  yambda-5b/
    interactions.parquet
    embeddings.parquet
```

For backward compatibility, the first 50m run automatically extracts these two
files from `dataset.zip` in the repository root when the cache is empty. The
public Kaggle Yambda archives currently have a different schema, so a remote
dataset reference must be agreed on before automatic downloading can safely be
enabled.

The same `PipelineConfig` and `PreparedInteractions` objects are used by every
experiment, so item mapping and train/eval boundaries are not reimplemented in
individual scripts.

## Pluggable SID evaluation

A SID implementation derives from `ISidPipeline`, accepts the prepared item
feature frame in `train`, and appends integer `sid_0`, ..., `sid_n` columns in
`update`. Load a class or zero-argument factory from a Python module or file:

```python
import polars as pl

from sids_rank_vs_generate.sids import ISidPipeline


class TrainMySid(ISidPipeline):
    def train(self, df: pl.DataFrame) -> None:
        self.model.fit(df["embed"])

    def update(self, df: pl.DataFrame) -> pl.DataFrame:
        codes = self.model.transform(df["embed"])
        return pl.DataFrame(
            {"item_id": df["item_id"], "sid_0": codes[:, 0], "sid_1": codes[:, 1]}
        )
```

```bash
python run.py \
  --dataset-size 50m \
  --sid experiments/my_sid.py:TrainMySid
```

The SID is required by the common runner. SASRec remains a separate baseline
because it does not produce semantic IDs.

The runner writes all machine-generated artifacts below one directory. For the
command above, the fixed layout is:

```text
reports/experiments/50m/TrainMySid/<UTC-run-id>/
  metrics.json
  ranking/   # metrics, plots, or other ranking artifacts when enabled
  candgen/   # metrics, plots, or other candidate-generation artifacts when enabled
```

Only `--reports-dir` changes the root. The generated UTC run id prevents repeated
experiments from overwriting each other. The runner does not generate a
narrative Markdown report: experiment-specific interpretation is written
manually in `reports/experiments/`.

The runner first restricts interactions to items covered by the supplied item
feature cache, then applies the usual core filter, chronological split, and
one shared dense item mapping. This prevents downstream examples from referring
to items that have no SID. Ranking and candidate-generation use fixed common
implementations and are enabled independently by flags:

```bash
python run.py \
  --dataset-size 50m \
  --sid experiments/my_sid.py:TrainMySid \
  --ranking \
  --candgen \
  --reports-dir reports/experiments
```

The concrete ranking and candidate-generation implementations still live in
their model branches. Until they are merged into the common stage registry,
enabling the corresponding flag fails with an explicit "stage is not connected"
error. A SID-only smoke run works without either flag.

## SASRec baseline

Run the baseline with its documented defaults:

```bash
python experiments/sasrec_baseline.py \
  --interactions /path/to/interactions.parquet \
  --metrics-json /tmp/sasrec_baseline.json
```

For a short CPU execution check:

```bash
python experiments/sasrec_baseline.py \
  --interactions /path/to/interactions.parquet \
  --epochs 1 \
  --device cpu
```

All model and data parameters are CLI options. The experiment output records the
effective config, preprocessing summary, per-epoch losses, and hit rate, recall,
NDCG, and coverage. The current experiment report is available at
[`reports/experiments/sasrec_baseline.md`](reports/experiments/sasrec_baseline.md).

## Verification

Run the repository smoke suite from the root:

```bash
python -m unittest discover -s tests -t . -v
```

The suite exercises preprocessing, the `run.py` command, deterministic SASRec
training/evaluation, metric output, and notebook syntax/output integrity. The
full homework quality run still requires the course `tests.py`, Yambda data, and
a GPU runtime; its outputs should not be committed to the notebook.
