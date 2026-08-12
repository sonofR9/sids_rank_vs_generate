# Correlation Figure Reports

This directory contains immutable, unpacked analysis outputs. Each report keeps
its figures (PNG/PDF/SVG), source tables, analysis metadata, and execution log
together so results can be inspected without access to MLflow.

## Layout

- `generation/logq_rqvae_sid_quality_recall/`: RQ-VAE + LogQ-only Recall@100
  analysis.
- `generation/pure_sid_logq_recall/`: mentor-requested pure SID LogQ
  Recall@100 export. It contains only `sid_logq_mode=standard`, with item
  LogQ disabled and auxiliary item loss equal to zero.
- `generation/poster_raw_sid_2026-08-10_2122/`: archived raw-SID poster
  generation snapshot from `sid_quality_correlations.zip`.
- `generation/poster_raw_sid_2026-08-10_2242/`: later raw-SID poster
  generation snapshot previously stored under `reports/poster/sid_quality_recall`.
- `generation/poster_raw_sid_with_plum_balance_2026-08-12/`: updated poster
  generation export including PLUM and BalanceRQKMeans. This report is produced
  on the GPU server by `scripts/analyze_generation_sid_quality.py`.
- `ranking/sid_quality_correlations/`: DCNv2 and TransAct correlations against
  SID quality metrics.

The two 2026-08-10 raw-SID snapshots are retained as separate historical
exports: they share the same experiment scope but contain non-identical source
tables and rendered figures. Compressed source archives are intentionally not
committed after their contents have been unpacked here.

## Reproduction

The analysis scripts in `scripts/` recreate the reports from MLflow. The
generation export with PLUM/Balance additionally reads the local run reports
and cached embeddings to backfill SID metrics that were not logged by the
original parent runs.
