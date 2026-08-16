# Downstream Results

Results on Yambda-50M. Relative changes are computed as
`(method / baseline - 1) * 100%`. Ranking rows use the same source tables as
the plots; the table shows the strongest comparable `L=3`, `C=256` SID result
for each model.

## Ranking

The relative change for each model is measured against its own item-only
baseline. All comparable SID rows are below baseline.

| Model | Method | Pairwise accuracy | vs item-only baseline |
| --- | --- | ---: | ---: |
| DCNv2 | DCNv2 item-only (baseline) | 0.5491444468 | -- |
| DCNv2 | RQVAE (ECM) | 0.5360032320 | -2.39% |
| TransAct | TransAct item-only (baseline) | 0.7616394758 | -- |
| TransAct | PLUM | 0.6376013756 | -16.29% |

ROC-AUC values are available in the separate ranking correlation figures; they
are not mixed into this table because the full-catalog TransAct source used by
the combined plots contains pairwise accuracy only.

## Generation

Generation values are Recall@100. The reference baseline is
SASRec-Audio-Uniform. The history-concat control concatenates frozen pretrained
item embeddings with learned item-id embeddings at every history position.
SID configurations use `L=2`, `C=256`.

| Method | Recall@100 | vs baseline |
| --- | ---: | ---: |
| SASRec-Audio-Uniform (baseline) | 0.1230292144 | -- |
| SASRec history-concat | 0.1281415231 | **+4.16%** |
| RQVAE-token | 0.1277876896 | **+3.87%** |
| RQVAE + item LogQ | 0.1413813459 | **+14.92%** |
| UAQ30 v1 capacity (`L=4`, `C=256`) | 0.0664585192 | -45.98% |

## Sources

- DCNv2 metrics: [`ranking_points.csv`](../figures/all_methods_sid_quality/ranking_points.csv)
- TransAct metrics: [`transact_ranking_points.csv`](../figures/all_methods_sid_quality/transact_ranking_points.csv)
- Generation metrics: [`generation_points.csv`](../figures/all_methods_sid_quality/generation_points.csv)
- UAQ30 source rows: [`user_aware_generation_runs.csv`](../figures/generation/user_aware_recall/user_aware_generation_runs.csv)
- SASRec baseline metrics: [`sasrec_yambda_50m_audio_uniform_seed42.json`](../experiments/sasrec/metrics/sasrec_yambda_50m_audio_uniform_seed42.json)
- SASRec history-concat metrics: [`sasrec_yambda_50m_audio_history_concat_seed42.json`](../experiments/sasrec/metrics/sasrec_yambda_50m_audio_history_concat_seed42.json)
