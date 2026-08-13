# Extended SID metrics with fixed TransAct results

This is a self-contained copy of `all_methods_sid_quality_fixed_transact` with
the newer SID metrics recalculated from the existing `semantic_ids.parquet`
files. The source report directory is left unchanged.

- `sid_metrics_calculated.csv` records metric values and parquet provenance for
  all 44 variant rows (36 unique SID parquet contents).
- `generation_points.csv`, `ranking_points.csv`, and `transact_metrics.csv`
  retain their original columns and add 44 recalculated SID metric columns.
- `extended_sid_metrics/` contains the 44 new SID metrics plotted against
  Recall@100, DCNv2 ROC-AUC, and TransAct pairwise accuracy in PNG, PDF, and SVG
  formats.
- `correlation_heatmaps/` contains Pearson and Spearman heatmaps against
  Recall@100, pairwise accuracy, and ROC-AUC targets. Undefined metrics and
  metrics with fewer than 20 complete pairs are excluded; separate files cover
  all runs and runs at or below the Recall@100 baseline.
- `extended_sid_metrics_metadata.json` records the input paths, compact item-ID
  alignment, metric names, and output counts.
- The original plots and CSV files are retained so this directory can be used
  as a complete report snapshot.

The compact embedding IDs are one-based, while these SID parquet item IDs are
zero-based. The calculation runner detects and records the resulting `-1`
embedding ID offset.

Reproduce the report from the repository root with:

```bash
python scripts/calculate_and_plot_extended_sid_metrics.py
python scripts/plot_extended_sid_quality_correlation_heatmaps.py
```

Use `--embeddings`, `--runs-root`, `--source-dir`, or `--output-dir` to override
the corresponding default paths.
