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

## Shared preprocessing

Prepare `uid`, `item_id`, and `timestamp` interactions with the project defaults
(5-core item filtering and a final 7-day evaluation split):

```bash
python run.py --interactions /path/to/interactions.parquet
```

To persist a machine-readable summary outside the repository:

```bash
python run.py \
  --interactions /path/to/interactions.parquet \
  --summary-json /tmp/sids_preprocessing.json
```

Use `python run.py --help` to inspect the core threshold, split duration, seed,
and output options. The same `PipelineConfig` and `PreparedInteractions` objects
are used by experiments, so item mapping and train/eval boundaries are not
reimplemented in individual scripts.

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
  --interactions /path/to/interactions.parquet \
  --embeddings /path/to/embeddings.parquet \
  --sid experiments/my_sid.py:TrainMySid
```

By default the complete JSON and concise Markdown summary are written to
`reports/experiments/TrainMySid.json` and
`reports/experiments/TrainMySid.md`. Use `--run-name`, `--reports-dir`,
`--metrics-json`, or `--report-md` to organize hyperparameter runs explicitly.

The runner first restricts interactions to items covered by the supplied item
feature parquet, then applies the usual core filter, chronological split, and
one shared dense item mapping. This prevents downstream examples from referring
to items that have no SID. Ranking and candidate-generation stages are
independently optional plugins implementing `IEvaluationStage`:

```bash
python run.py \
  --interactions /path/to/interactions.parquet \
  --embeddings /path/to/embeddings.parquet \
  --sid experiments/my_sid.py:TrainMySid \
  --ranking-evaluator package.ranking:RankingEvaluator \
  --candgen-evaluator package.candgen:CandgenEvaluator \
  --metrics-json /tmp/my_sid.json \
  --report-md reports/experiments/my_sid.md
```

Evaluator implementations live in the model-specific branches. Keeping the
stages injectable lets a SID run ranking, candidate generation, or both without
changing the common entry point.

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
