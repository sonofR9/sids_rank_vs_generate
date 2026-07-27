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
experiments. A SID implementation derives from `ISidPipeline` and is loaded
with `--sid`:

```bash
python run.py \
  --dataset-size 50m \
  --sid experiments/my_sid.py:TrainMySid
```

Baselines that do not produce semantic IDs, such as SASRec, are run separately.
The pluggable ranking and candidate-generation stages remain available through
the common SID runner.

## Train SASRec

The first run downloads the 7 MB Yambda-50M likes file and prepares the compact
embedding cache. Later runs reuse both caches.

```bash
python run.py \
  --epochs 10 \
  --batch-size 128 \
  --metrics-json reports/experiments/sasrec_yambda_50m.json
```

The SID is required by the common runner. Baselines that do not produce
semantic IDs, such as SASRec, are run separately.
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
