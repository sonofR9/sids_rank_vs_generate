# Pure SID LogQ Recall@100

This immutable export contains only the pure SID-level LogQ experiment:

- SID correction: `score(SID) - log q(SID)`;
- `sid_logq_mode=standard`, `sid_logq_beta=1.0`;
- item-level `logq_mode=none`;
- `auxiliary_item_loss_weight=0.0`.

The two completed configurations are `L=2, C=256` (Recall@100 `0.12691854`,
MLflow run `75deeabc883e4c8c8be7a1f8a3bbb332`) and `L=3, C=256`
(Recall@100 `0.12101535`, MLflow run `81649a5f46ac4b94a726496911a45f08`).

`plots/` contains each SID-quality versus Recall@100 scatter figure in PNG,
PDF, and SVG. The CSV files retain source runs, metric coverage, correlation
calculations, point legends, and excluded runs. With only two points, any
reported Pearson or Spearman value of `+/-1` is descriptive only and is not a
statistical conclusion.

Reproduce with:

```bash
python scripts/analyze_logq_rqvae_sid_quality_recall.py \
  --tracking-uri http://51.250.47.231:5000 \
  --logq-experiments sid-comparison \
  --quality-experiments sid-quality-only-full-v2 \
  --pure-sid-logq-only \
  --output-dir reports/figures/generation/pure_sid_logq_recall
```
