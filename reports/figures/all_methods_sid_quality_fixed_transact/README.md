# Fixed TransAct metrics

This is a copy of `all_methods_sid_quality` with only the TransAct measurements
updated from the completed 44-variant sweep. The original report directory is
left unchanged.

- `transact_metrics.csv` contains the 44 source measurements.
- `ranking_points.csv` contains the updated pairwise accuracy and ROC-AUC.
- `transact/` contains the 12 regenerated TransAct-versus-SID plots in PNG,
  PDF, and SVG formats.
- `generation_points.csv`, `recall100/`, and `dcnv2/` are copied from the
  original report for a self-contained snapshot.

Reproduce the report from the repository root with:

```bash
python scripts/plot_all_generation_methods_fixed_transact.py
```
