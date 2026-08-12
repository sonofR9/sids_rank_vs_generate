# SID Quality Figure Reports

All committed visual reports live here. Every report contains rendered figures
(PNG/PDF/SVG), source CSV tables, and metadata. Checkpoints, raw training
artifacts, tmux logs, and compressed delivery archives are deliberately kept
out of Git.

## Generation

- `generation/rqvae_item_logq_recall/`: prior RQ-VAE item-level LogQ
  Recall@100 analysis; it can include auxiliary item loss configurations.
- `generation/rqvae_pure_sid_logq_recall/`: pure SID-level correction
  `score(SID) - log q(SID)`, with item LogQ disabled and auxiliary item loss
  equal to zero. The current report contains `L=2,C=256` and `L=3,C=256`.
- `generation/raw_sid_recall/`: final raw-SID Recall@100 report. It supersedes
  the intermediate snapshot and retains the additional raw-run matching table.
- `generation/poster_plum_balance_recall/`: poster generation
  Recall@100 export including PLUM and BalanceRQKMeans.

## Ranking

- `ranking/`: DCNv2 and TransAct Pairwise Accuracy and ROC-AUC correlations
  against SID quality metrics. The `dcnv2/` and `transact/` directories hold
  model-specific figures; shared CSV summaries live directly here.

## Reproduction

`scripts/analyze_generation_sid_quality.py`,
`scripts/analyze_logq_rqvae_sid_quality_recall.py`, and
`scripts/analyze_ranking_sid_quality.py` recreate their corresponding reports
from MLflow. The PLUM/Balance generation export additionally uses local run
reports and cached embeddings to backfill metrics absent from the original
parent runs.
