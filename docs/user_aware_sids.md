# User-aware Semantic IDs

## Research question

The experiment tests whether Semantic IDs that include a train-only user-item
signal are more useful than content-only Semantic IDs for ranking and
autoregressive candidate generation.

The SID construction method is the experimental variable. Dataset preparation,
time boundaries, random seed, downstream architectures, training budgets, and
evaluation code must remain fixed across methods.

## Representations

Every catalog item first receives a global content SID:

```text
item -> global SID
```

A user-aware pipeline additionally assigns a personalized SID to a catalog pair:

```text
(frozen train-only user state, item) -> user SID
```

The MVP deliberately uses a static user state within a run. It is fitted only
from interactions strictly before the validation boundary and is frozen for
validation and test. A future contextual variant may replace it with a
prefix-dependent state `(history before t, item) -> user SID`, but that is a
different experiment and must not silently share results with the static MVP.

The generation treatment predicts a user SID and maps it back to a global SID:

```text
(user, history) -> user SID
(user, user SID) -> global SID
global SID -> catalog item
```

The direct-generation control predicts the global SID without the intermediate
personalized representation:

```text
(user, history) -> global SID -> catalog item
```

## Leakage-safe data protocol

The common data preparation must expose three chronological downstream splits:

1. `train`: fit user aggregation, SID codebooks, mapper, and downstream models;
2. `validation`: checkpoint selection and early stopping only;
3. `test`: final metrics only.

The item core and catalog mapping must be derived from train data. Validation
and test are then restricted to that train-known catalog. User aggregation and
SID codebooks must never read validation or test rows.

The MVP permits a transductive train-only representation: an interaction may
contribute to the frozen user state used by other training examples from the
same train split. It does not permit validation/test leakage. A stricter
prefix-causal or leave-one-out training protocol is a separate ablation.

The runner must record the split boundaries, row counts, and a deterministic
fingerprint of the interactions used to fit the user-aware pipeline.

## MVP user signal

For user `u`, aggregate the normalized content embeddings of train items:

```text
p_u = normalize(mean(e_i for (u, i) in train))
```

The first implementation uses one contribution per distinct `(user, item)`
pair. Later configurations may add frequency, recency decay, matrix
factorization, or graph embeddings without changing the pipeline contract.

For a pair `(u, i)`, form the feature vector:

```text
z_ui = [e_i, p_u, e_i * p_u, abs(e_i - p_u)]
```

An RQ-KMeans codebook fitted on train pairs quantizes `z_ui` into
`usid_0, ..., usid_n`. Unknown users use the normalized global train-user mean;
items outside the train-known catalog are rejected.

## Common pipeline contract

Content-only SID implementations continue to implement `ISidPipeline` and
produce one `sid_<level>` path per item.

User-aware implementations implement `IUserAwareSidPipeline`:

```python
train(UserAwareSidTrainingData) -> None
update(unique user-item pairs) -> uid, item_id, usid_0, ..., usid_n
```

`UserAwareSidTrainingData` contains only train interactions, item features,
global item SIDs, and the exclusive fit-end timestamp. The runner validates the
boundary before calling a pipeline.

## Integration with the main experiment API

The user-aware implementation extends the existing runner; it does not replace
it and does not introduce a second experiment entry point.

The existing `ExperimentConfig.sid_pipeline: ISidPipeline` remains required and
continues to construct the global item SID. Existing content-only experiment
files therefore remain valid without modification. An optional user-aware
configuration is added alongside it:

```python
ExperimentConfig(
    catalog_from_fit=True,
    sid_pipeline=global_sid_pipeline,
    user_sid_pipeline=user_sid_pipeline,
    training=TrainingConfig(
        generation=GenerationConfig(
            mapper=UserSidMapperConfig(kind="transformer"),
        ),
    ),
)
```

`run_sid_experiment` first executes the unchanged global SID path. If
`user_sid_pipeline` is configured, it then fits the auxiliary pair pipeline from the
prepared train split and the already constructed global SIDs. The fitted
pipeline is passed through the existing `EvaluationContext` to the fixed
ranking and generation stages.

Personalized SIDs are not represented as one row per catalog item and therefore
must not be forced through `ISidPipeline.update(item_features)`. They are
computed lazily for the unique `(uid, item_id)` pairs required by each stage.
This avoids materializing the full user-by-catalog Cartesian product.

The common `PreparedInteractions` contract exposes fit, validation, and test.
Mapper and downstream model weights are learned from fit, validation drives
model selection, and test is reserved for final reporting. The old content-only
configuration remains accepted and is routed through the same split so
baseline and treatment see identical rows.

## Mapper

The reproducible baseline mapper is a train-only frequency model. It estimates the
most frequent global SID for `(user, SID_u)`, falls back to the train-wide
`SID_u -> SID` distribution for unseen user-specific keys, and finally falls
back to the global train distribution.

The learned option is a small conditional encoder-decoder Transformer. Its
encoder receives the frozen train-only dense user profile plus one embedding
for every `SID_u` level. Its autoregressive decoder predicts the global SID
levels. Mapper weights are fitted only from fit pairs, validation is used for
early stopping, and test is used only for final metrics. Decoding is masked
with a prefix trie constructed from catalog global SIDs, so the model cannot
return a token combination that does not resolve to a catalog path.

Both implementations expose the same top-k interface and are selected with
`UserSidMapperConfig(kind="frequency")` or
`UserSidMapperConfig(kind="transformer")`. The Transformer checkpoint is saved
as `generation/best_mapper.pt`.

Report mapper token accuracy, exact global-SID accuracy, and top-k global-SID
accuracy separately from end-to-end item retrieval. This distinguishes a weak
user-SID generator from a weak mapper.

## Ranking protocol

For the user-aware treatment, each candidate row receives `usid_<level>` from
the frozen user-aware pipeline. DCNv2 and TransAct consume these candidate
features while all dense, artist, album, and history inputs remain unchanged.
The content-only control uses the global `sid_<level>` columns.

## Metrics

Every method must report:

- global SID statistics: utilization, coverage, collisions, and PAS;
- user-SID pair statistics on unique pairs: utilization and collision rate;
- per-user uniqueness and collision distribution;
- variation of one item's user SID across users;
- mapper token/exact/top-k metrics;
- ranking ROC-AUC, PR-AUC, logloss, and pairwise accuracy;
- generation token accuracy, SID exact match, item Recall@K, MRR@K, and
  catalog coverage;
- runtime, peak memory when available, and artifact paths.

Final claims use only test metrics. Validation metrics select checkpoints and
are stored separately.

## Comparison matrix

At minimum, run the same downstream configuration for:

1. content RQ-KMeans -> direct global-SID generation;
2. content RQ-VAE -> direct global-SID generation;
3. user-aware RQ-KMeans -> mapper -> global SID;
4. user-aware RQ-KMeans with the user component removed, as an implementation
   control equivalent to content-only pair encoding.

RQ-VAE over user-aware pair features is a follow-up after the RQ-KMeans MVP is
validated.

## Definition of done

- The user-aware pipeline is configured through the common experiment config.
- Train-only aggregation and frozen validation/test behavior are covered by
  regression tests.
- Content-only experiments still run unchanged.
- Direct and personalized generation share data, seed, architecture, and
  training budget.
- Ranking accepts global or personalized candidate SID features.
- One command writes a unified `metrics.json`; optional MLflow logging consumes
  the same result without changing experiment behavior.
- RQ-KMeans, RQ-VAE, and the user-aware MVP have reproducible test reports.
