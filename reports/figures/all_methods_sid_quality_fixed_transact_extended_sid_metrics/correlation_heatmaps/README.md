# Extended SID correlation heatmaps

Pearson and Spearman correlations are pooled across all available SID methods and configurations. The five rows are generation Recall@100, then pairwise accuracy and ROC-AUC for DCNv2 and TransAct. Candidate columns contain all 12 original and 44 newly calculated SID metrics.

Each cell reports the correlation coefficient and complete-pair count. Metrics are removed if any target correlation is undefined or uses fewer than 20 complete pairs; 38 of 56 metrics remain. Colors span the observed minimum and maximum independently for each correlation type, with zero at the neutral midpoint when the range crosses zero.

Additional files ending in `_below_recall_baseline` use only method, level, and codebook configurations whose generation Recall@100 is less than or equal to 0.113. The same keys filter the ranking rows, and 31 metrics pass the independent coverage filter for this subset.

Outputs are written as PNG, PDF, and SVG, with long-form correlations and matrix CSV files alongside them.
