# Correlation Figure Reports

This directory contains immutable, unpacked analysis outputs. Each report keeps
its figures (PNG/PDF/SVG), source tables, analysis metadata, and execution log
together so results can be inspected without access to MLflow.

## Layout

- `generation/logq_rqvae_sid_quality_recall/`: RQ-VAE + LogQ-only Recall@100
  analysis.
- `generation/poster_raw_sid/`: final raw-SID poster generation report,
  previously stored under `reports/poster/sid_quality_recall`.
- `generation/poster_raw_sid_with_plum_balance/`: updated poster
  generation export including PLUM and BalanceRQKMeans. This report is produced
  on the GPU server by `scripts/analyze_generation_sid_quality.py`.
- `generation/user_aware_recall/`: completed full-catalog UAQ30 generation
  points. These are included only in Recall@100 panels because the runs did not
  execute the DCNv2 or TransAct ranking stages.
- `ranking/sid_quality_correlations/`: DCNv2 and TransAct correlations against
  SID quality metrics. Both model families include their dashed item-only
  baselines: DCNv2 (`0.5491444468` pairwise accuracy, `0.8093270063` ROC-AUC)
  and TransAct (`0.7616394758` pairwise accuracy, `0.9367052913` ROC-AUC).
- `ranking/transact_sid_token_history/`: compact full-catalog source table for
  the TransAct SID-token protocol, including PLUM, BalanceRQKMeans, item LogQ,
  and pure SID LogQ runs.
- `all_methods_sid_quality/`: combined two-panel figures. For each SID metric,
  it contains Recall@100, DCNv2 Pairwise Accuracy, and TransAct Pairwise
  Accuracy views across all compatible method families, including both LogQ
  variants and compatible UAQ30 user-aware generation runs. PLUM,
  BalanceRQKMeans, and UAQ30 use the 33,148-item catalog.
  DCNv2 and TransAct panels include their respective item-only baseline lines;
  generation panels include both SASRec-Audio-Uniform and the pretrained
  history-concat control.

The intermediate raw-SID snapshot from `sid_quality_correlations.zip` is not
kept separately: the later export contains the final figures and one additional
raw-run matching table. Compressed source archives are intentionally not
committed after their contents have been unpacked here.

The standalone downstream results table is available at
[`reports/tables/downstream_results.md`](../tables/downstream_results.md), with
the same rows in CSV form next to it.

## Reproduction

The analysis scripts in `scripts/` recreate the reports from MLflow. The
generation export with PLUM/Balance additionally reads the local run reports
and cached embeddings to backfill SID metrics that were not logged by the
original parent runs.

`scripts/plot_all_generation_methods.py` creates the combined figures from the
committed CSVs. It recognizes both this layout and the flattened layout on
current `main`. DCNv2 panels use the standard ranking export, while TransAct
panels use the full SID-token table so the extension and LogQ methods are not
dropped. The figures can be rendered from the committed CSVs without an MLflow
installation; MLflow is imported only for the query-and-export path.
