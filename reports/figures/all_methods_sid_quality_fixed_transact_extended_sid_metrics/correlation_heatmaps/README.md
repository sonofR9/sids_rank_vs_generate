# Extended SID correlation heatmaps

Pearson and Spearman correlations are pooled across all available SID methods and configurations. The five rows are generation Recall@100, then pairwise accuracy and ROC-AUC for DCNv2 and TransAct. The 56 columns contain all 12 original and 44 newly calculated SID metrics.

Each cell reports the correlation coefficient and complete-pair count. DCNv2 uses the 28 configurations available in the combined report; generation and TransAct use all 44. Gray cells are undefined because the SID metric is constant or has fewer than two complete pairs.

Outputs are written as PNG, PDF, and SVG, with long-form correlations and matrix CSV files alongside them.
