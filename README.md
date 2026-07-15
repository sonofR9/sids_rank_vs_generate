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
