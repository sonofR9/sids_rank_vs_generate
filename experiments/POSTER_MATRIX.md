# Poster experiment matrix

The shared config is `experiments/poster_sid_config.py`; the launcher is
`scripts/run_poster_matrix.sh`. All runs use seed 42, the complete Yambda-50M
interaction sources and validation examples, Recall@100, and skip the unrelated
ranking stage. CUDA training automatically uses BF16 where supported and FP16
with gradient scaling on older GPUs such as V100.

Ranking-only reruns use `scripts/run_poster_ranking_matrix.sh`. Its
`ranking_none_all` set mirrors the 28 completed raw-SID (`none`) generation
configurations, skips generation, and uses the accelerated ranking schedules
from the current `main` branch.

The matrix is deliberately phased:

- `phase_a`: L3/C256 anchors plus RQVAE nearest/token/RRS/ECM. Existing
  L3/C256 collision-token runs for RQKMeans and both R3VAE initializations are
  reused from MLflow and are not retrained.
- `phase_b`: L2/C256 and L4/C256 raw-SID endpoints for all four families.
- `phase_c`: L3/C128 raw-SID codebook-size control for all four families.

RQ-VAE checkpoints are shared across assignment and collision-resolution
variants. R3-VAE checkpoints are shared across collision-resolution variants,
but kept separate for spherical and RQKMeans initialization.

## GPU server

After the branch is pushed, update the server checkout and environment:

```bash
cd /home/k.ryadinskiy/projects/sids_rank_vs_generate
git fetch origin
git switch poster-experiments
git pull --ff-only
source venv/bin/activate
pip install -e '.[poster]'
```

Run one phase at a time on a single GPU:

```bash
tmux new-session -d -s poster-a \
  "cd /home/k.ryadinskiy/projects/sids_rank_vs_generate && \
   source venv/bin/activate && \
   RUN_SET=phase_a GPU=0 bash scripts/run_poster_matrix.sh \
   2>&1 | tee reports/poster-phase-a.log"
tmux attach -t poster-a
```

After phase A, detach with `Ctrl-b d`. If the collision range is useful, run
the next phases by replacing `poster-a`/`phase_a` with `poster-b`/`phase_b` and
then `poster-c`/`phase_c`.

Export compatible historical and new runs, aggregate multiple seeds, and
render PNG/PDF/SVG:

```bash
python scripts/export_poster_results.py \
  --tracking-uri http://51.250.47.231:5000 \
  --experiment-name sid-comparison \
  --output-dir reports/poster
```

The raw table is `poster_results.csv`; the deduplicated seed summary is
`poster_results_summary.csv`. The plot uses raw pre-resolution ICR on the
x-axis, so collision tokens and PSI methods are compared at the same original
collision regime.
