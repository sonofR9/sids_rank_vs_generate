# RQKMeans Recall@100 experiments

The default runner is now the main `L3/C1024`, no-collision-resolution
candidate.  It uses catalog-constrained beam search, 64 history events,
content and semantic history initialization, an auxiliary in-batch item loss,
and reranks the best 100 items from 512 valid generated candidates.

Run it from the repository root:

```bash
python -m experiments.sid_comparison.run
```

If the remote MLflow server is unavailable, append
`--mlflow-tracking-uri file:./mlruns`.

The checkpoint and JSON result are written below:

```text
reports/experiments/rqkmeans_C1024-1024-1024_without_collision_resolution_recall_v2/
```

## Recommended order

1. Run the default candidate above.  Compare both `recall@100` and
   `candidate_recall@512`.  The latter is the generator's upper bound before
   the item reranker chooses 100 candidates.
2. Measure how much comes from the pure generator:

   ```bash
   python -m experiments.sid_comparison.run \
     --generative-only \
     --experiment-tag recall_v2_generator
   ```

3. If `candidate_recall@512` is strong but final recall is below 0.14, sweep
   the auxiliary/reranking strength:

   ```bash
   python -m experiments.sid_comparison.run \
     --auxiliary-item-loss-weight 0.1 \
     --rerank-weight 0.25 \
     --experiment-tag aux01_rank025

   python -m experiments.sid_comparison.run \
     --auxiliary-item-loss-weight 0.4 \
     --rerank-weight 1.0 \
     --experiment-tag aux04_rank10
   ```

4. If `candidate_recall@512` itself is the bottleneck, try easier tapered
   token spaces and a longer context:

   ```bash
   python -m experiments.sid_comparison.run \
     --codebook-sizes 512,256,128 \
     --experiment-tag tapered_512_256_128

   python -m experiments.sid_comparison.run \
     --codebook-sizes 256,128,64 \
     --experiment-tag tapered_256_128_64

   python -m experiments.sid_comparison.run \
     --max-history-len 100 \
     --batch-size 256 \
     --experiment-tag history100
   ```

5. The following controls isolate the two history signals and the path
   constraint:

   ```bash
   python -m experiments.sid_comparison.run \
     --no-content-init \
     --experiment-tag no_content

   python -m experiments.sid_comparison.run \
     --no-semantic-history \
     --experiment-tag no_semantic_history

   python -m experiments.sid_comparison.run \
     --generative-only \
     --unconstrained-beam \
     --num-beams 150 \
     --max-history-len 16 \
     --batch-size 1024 \
     --no-content-init \
     --no-semantic-history \
     --experiment-tag legacy_shape_control
   ```

Do not launch the full matrix at once.  The default run and its
`candidate_recall@512` diagnostic determine whether the next GPU budget should
go to the reranker or to generation.
