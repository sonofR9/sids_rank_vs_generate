# All-method SID quality correlation heatmaps

The heatmaps pool every available SID method and configuration from
`all_methods_sid_quality`. Columns are SID quality metrics. Rows separate the
generative target (`Recall@100`) from four ranking targets: pairwise accuracy
and ROC-AUC for both `DCNv2` and `TransAct`.

Each cell shows the correlation coefficient and the number of complete pairs
used for that coefficient. Pearson, Spearman, and Kendall tau are calculated independently
after dropping rows missing either metric. Gray `NA` cells indicate a constant
input or fewer than two complete pairs. Each heatmap uses its observed minimum
and maximum as the color limits, with zero kept at the neutral midpoint.
`Prefix Coherence@3` is marked with an asterisk on the X axis.
Only configurations with generation Recall@100 <= 0.113 are included. The same configuration keys filter the ranking rows.

Outputs:

- `spearman_heatmap.*`: Spearman rank correlation.
- `pearson_heatmap.*`: Pearson linear correlation.
- `kendall_heatmap.*`: Kendall rank correlation.
- `correlation_heatmaps.*`: all three heatmaps in one figure.
- `correlations.csv`: long-form coefficients, pair counts, and undefined reasons.
- `pearson_matrix.csv`, `spearman_matrix.csv`, `kendall_matrix.csv`, `num_points_matrix.csv`: compact matrices.

Undefined cells: 5 of 60.
