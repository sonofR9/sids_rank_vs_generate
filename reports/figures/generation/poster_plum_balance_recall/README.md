# Poster Generation With PLUM and Balance

This directory receives the final generation-side poster export produced on the
GPU server after adding three PLUM and three BalanceRQKMeans configurations.

Expected contents are the PNG/PDF/SVG `recall100_vs_*` figures together with
`matched_generation_runs.csv`, `correlations.csv`, `analysis_metadata.json`,
and `local_sid_metric_backfill.csv`.

The report is intentionally separate from the 2026-08-10 snapshots because it
contains 36 generation runs and derives missing PLUM/Balance SID metrics from
the saved semantic IDs and cached catalog embeddings.
