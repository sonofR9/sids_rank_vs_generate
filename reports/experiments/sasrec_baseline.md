# SASRec baseline

## Dataset and preprocessing

The experiment expects Yambda interactions from `interactions.parquet` and uses
the canonical columns `uid`, `item_id`, and `timestamp`. It calls the shared
`sids_rank_vs_generate.data` preprocessing rather than maintaining an
experiment-specific copy.

The default preparation is:

- keep items with at least 5 interactions;
- map retained item ids once to the dense range `[0, catalog_size)`;
- sort interactions chronologically;
- use the final 7 days as the evaluation split and all earlier interactions as
  the training split.

Training examples are next-item prefixes truncated to the final 100 events.
Evaluation uses the final available training history for each user who has at
least one event in the evaluation split.

## Default parameters

| Parameter | Value |
| --- | ---: |
| Embedding dimension | 64 |
| Maximum sequence length | 100 |
| Transformer layers | 2 |
| Attention heads | 2 |
| Dropout | 0.1 |
| Batch size | 128 |
| Epochs | 10 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Evaluation cutoff | 100 |
| Seed | 42 |

The model uses a causal Transformer encoder, shared input/output item
embeddings, full-catalog cross-entropy, and the last valid sequence state for
catalog scoring.

## Command

From the repository root:

```bash
python experiments/sasrec_baseline.py \
  --interactions interactions.parquet \
  --metrics-json reports/experiments/sasrec_baseline.json
```

Every default can be overridden from the command line; run
`python experiments/sasrec_baseline.py --help` for the complete list.

## Metrics

The experiment reports hit rate, recall, NDCG, and catalog coverage at the
configured `topk`. Its JSON output also records preprocessing counts, the full
configuration, device, number of training examples and evaluation users, and
the loss for each epoch.

## Results

A full Yambda baseline run has not been executed in this environment because
the dataset is intentionally not stored in the repository. Therefore no
full-dataset metric values are reported yet.

The automated synthetic CPU smoke run completed successfully. These values are
only an execution check and are not comparable to Yambda quality:

| Smoke value | Result |
| --- | ---: |
| Interactions / users / catalog | 8 / 2 / 3 |
| Train interactions / test interactions | 6 / 2 |
| Epochs / seed | 1 / 7 |
| Training loss | 1.0870763063 |
| Hit rate@2 | 0.0 |
| Recall@2 | 0.0 |
| NDCG@2 | 0.0 |
| Coverage@2 | 0.6666666667 |

The smoke experiment was repeated with the same seed and produced identical
losses and metrics.

## Run environment

- Date: 2026-07-15
- Platform: macOS 26.5.1, arm64
- Device: CPU
- Python: 3.14.5
- PyTorch: 2.11.0
- Polars: 1.42.1

## Comparison with the homework Transformer

Both implementations use a causal Transformer, the same chronological split,
and the same ranking metrics. A numerical comparison is not available: the
SASRec full-dataset run has not been performed, and notebook outputs are not
stored in Git. The comparison should be added only after running both models on
the same prepared data and evaluation cutoff.
