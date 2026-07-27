# SIDS rank vs generate

The repository contains the shared preprocessing and evaluation runner used by
SID experiments.
It also contains a SASRec baseline trained on the official Yambda-50M
**likes** split.
Semantic IDs are not created or used in this stage.

## Data

The single supported interaction source is:

```text
Hugging Face: yandex/yambda
file: flat/50m/likes.parquet
rows: 881,456
```

Yambda timestamps are integer bins of five seconds. The preprocessing uses a
5-core item filter and the final seven days as the test interval.

The official global audio-embedding file is 13.8 GB. The first run therefore
uses the repository's existing `embeddings.parquet` once to create a compact
Yambda-50M-only cache:

```text
~/.cache/sids_rank_vs_generate/datasets/yambda-50m/
  embeddings.parquet
```

That cache contains 170,787 liked tracks with L2-normalized 128-dimensional
audio vectors. The experiment first restricts likes to that catalog and then
applies the 5-core filter. With the current data this leaves:

```text
interactions: 634,642
users:          8,032
items:         33,148
train:        630,095
test:           4,547
```

The course Kaggle `embeddings_small.parquet` is intentionally not used: it
covers only 144,670 of the 181,304 tracks liked in Yambda-50M, versus 170,787
covered by the prepared official vectors.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Datasets, archives, checkpoints, and generated metric files are local artifacts
and must not be committed.

## Shared SID runner

The common runner selects the Yambda scale and keeps dataset paths, the 5-core
item filter, the final 7-day split, and the random seed shared by SID
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

Baselines that do not produce semantic IDs, such as SASRec, are run separately.

The runner writes all machine-generated artifacts below one directory:

```text
reports/experiments/50m/TrainMySid/<UTC-run-id>/
  metrics.json
  ranking/
  candgen/
```

Only `--reports-dir` changes the root. The generated UTC run id prevents repeated
experiments from overwriting each other. The runner first restricts interactions
to items covered by the supplied item feature cache, then applies the common core
filter, chronological split, and dense item mapping.

Ranking and candidate generation are enabled independently:

```bash
python run.py \
  --dataset-size 50m \
  --sid experiments/my_sid.py:TrainMySid \
  --ranking \
  --candgen \
  --reports-dir reports/experiments
```

Until their concrete implementations are merged into the common stage registry,
enabling an unconnected stage fails with an explicit error. A SID-only smoke run
works without either flag.

## Train SASRec

The first run downloads the 7 MB Yambda-50M likes file and prepares the compact
embedding cache. Later runs reuse both caches.

```bash
python run.py \
  --epochs 10 \
  --batch-size 128 \
  --metrics-json reports/experiments/sasrec_yambda_50m.json
```

The 128-dimensional normalized audio vectors initialize SASRec's item embedding
table and remain trainable by default. Use `--freeze-item-embeddings` to keep
them fixed.

Training uses sampled negatives instead of a full-catalog softmax. Exact top-k
evaluation is computed in catalog chunks, so the experiment does not allocate a
batch-by-catalog score matrix.

The default `sampled-uniform` objective draws negatives uniformly from the
catalog and reproduces the first full baseline. The `in-batch` objectives draw
from all non-padding history items and targets in the current batch. For each
example, candidates contributed by the same user, every item in its own prefix,
and every occurrence of its positive item id are excluded from the negative
pool. `in-batch-logq` additionally corrects only negative logits using base item
frequencies in the aggregate history-plus-target pools produced by all train
examples.

```bash
python run.py \
  --epochs 10 \
  --batch-size 128 \
  --training-objective in-batch \
  --training-negatives 100 \
  --metrics-json reports/experiments/sasrec_yambda_50m_audio_inbatch_user_mask_seed42.json
```

```bash
python run.py \
  --epochs 10 \
  --batch-size 128 \
  --training-objective in-batch-logq \
  --training-negatives 100 \
  --metrics-json reports/experiments/sasrec_yambda_50m_audio_logq_user_mask_seed42.json
```

These objectives keep the model, audio initialization, data split, seed, and
exact top-k evaluation unchanged. The 100-negative LogQ variant achieved the
best Recall@100; increasing its negative count to 2,048 removed that gain. See
`reports/experiments/sasrec_baseline.md` for the full controlled comparison.

Useful overrides:

```bash
python run.py \
  --epochs 1 \
  --device cpu \
  --training-negatives 50 \
  --eval-item-chunk-size 10000
```

To use explicitly prepared files:

```bash
python run.py \
  --interactions /path/to/likes.parquet \
  --embeddings /path/to/embeddings.parquet
```

## Verification

```bash
python -m unittest discover -s tests -t . -v
```

The tests cover chronological preprocessing, five-second timestamp handling,
content initialization, deterministic sampled-negative training, chunked
ranking evaluation, metric output, the shared runner contract, SID validation,
stage selection, and artifact output.
