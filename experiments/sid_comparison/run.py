import json
import math
import numbers
from dataclasses import replace
from pathlib import Path

import mlflow

from experiments.sid_comparison.config import (
    COLLISION_RESOLUTION_MODES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SID_Hyperparametrs,
    SID_Methods,
    make_transformer_config,
)
from experiments.rqkmeans.config_rqkmeans import RQKMeansConfig
from experiments.rqvae.config_plum_rqvae import PLUMRQVAEConfig
from experiments.rqvae.config_rqvae import RQVAEConfig
from sids_rank_vs_generate.data.sem_trans_dataset import (
    build_loaders,
    limit_examples,
    prepare_examples,
)
from sids_rank_vs_generate.datasets_for_sem_transform import load_yambda_data
from sids_rank_vs_generate.evaluation.collision_quality import (
    PAS,
    codebook_load_balance_and_coverage_and_utitlization,
)
from sids_rank_vs_generate.evaluation.sid_statistics import tests_semantic_ids
from sids_rank_vs_generate.models.semantic_transformer_model import (
    SemanticTransformerModel,
)
from sids_rank_vs_generate.sids.codebook_sizes import resolve_codebook_sizes
from sids_rank_vs_generate.sids.collision_resolution import (
    add_collision_resolving_sid,
    get_sid_cols,
)
from sids_rank_vs_generate.sids.sid_builder import build_semantic_ids
from sids_rank_vs_generate.training.semantic_trainer import train_semantic


def flatten_metrics(metrics, prefix=""):
    flattened = {}

    for name, value in metrics.items():
        metric_name = f"{prefix}/{name}" if prefix else str(name)
        metric_name = metric_name.replace("@", "_at_").replace(" ", "_")

        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, metric_name))
        elif (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            flattened[metric_name] = float(value)

    return flattened


def log_mlflow_metrics(metrics, step=None, prefix=""):
    metrics = flatten_metrics(metrics, prefix)
    if metrics:
        mlflow.log_metrics(metrics, step=step)


def build_raw_semantic_ids(
    method,
    embeddings,
    codebook_sizes,
    metric_logger=None,
):
    rq_num_levels = len(codebook_sizes)
    rqkmeans_config = None
    rqvae_config = None
    plum_config = None

    if method == "rqkmeans":
        rqkmeans_config = replace(
            RQKMeansConfig(),
            num_levels=rq_num_levels,
            codebook_sizes=codebook_sizes,
        )
    elif method == "rqvae":
        rqvae_config = replace(
            RQVAEConfig(),
            num_levels=rq_num_levels,
            codebook_sizes=codebook_sizes,
        )
    elif method == "plum":
        plum_config = replace(
            PLUMRQVAEConfig(),
            num_levels=rq_num_levels,
            codebook_sizes=codebook_sizes,
        )
    else:
        raise ValueError(f"Unknown SID method: {method}")

    raw_semantic_df = build_semantic_ids(
        method=method,
        embeddings=embeddings,
        rqvae_config=rqvae_config,
        rqkmeans_config=rqkmeans_config,
        plum_config=plum_config,
        metric_logger=metric_logger,
    )

    sid_metrics = {
        "sid_statistics": tests_semantic_ids(
            raw_semantic_df,
            num_levels=rq_num_levels,
        ),
        "PAS": PAS(raw_semantic_df, embeddings),
        "codebook": codebook_load_balance_and_coverage_and_utitlization(
            raw_semantic_df,
            codebook_sizes,
        ),
    }

    return raw_semantic_df, sid_metrics


def build_sid_to_items(semantic_df, item2idx):
    sid_cols = get_sid_cols(semantic_df)
    sid_to_items = {}

    for row in semantic_df.select(
        [*sid_cols, "item_id"]
    ).iter_rows(named=True):
        sid = tuple(row[column] for column in sid_cols)
        item_idx = item2idx[row["item_id"]]
        sid_to_items.setdefault(sid, []).append(item_idx)

    return sid_to_items


def run_single_experiment(
    method,
    codebook_sizes,
    data,
    raw_semantic_df,
    sid_metrics,
    base_transformer_cfg,
    metric_logger=None,
):
    rq_num_levels = len(codebook_sizes)
    resolve_collisions = base_transformer_cfg.resolve_collisions
    collision_mode = (
        "with_collision_resolution"
        if resolve_collisions
        else "without_collision_resolution"
    )
    experiment_name = (
        f"{method}_C{'-'.join(map(str, codebook_sizes))}"
        f"_{collision_mode}"
    )

    print(f"\nStarting {experiment_name}")

    output_dir = Path("reports/experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if resolve_collisions:
        semantic_df, max_collision_token = add_collision_resolving_sid(
            semantic_df=raw_semantic_df,
        )
        required_collision_codebook_size = max_collision_token + 1
        collision_codebook_size = (
            base_transformer_cfg.collision_codebook_size
            or required_collision_codebook_size
        )

        if collision_codebook_size < required_collision_codebook_size:
            raise ValueError(
                "collision_codebook_size must be at least "
                f"{required_collision_codebook_size}, got "
                f"{collision_codebook_size}"
            )

        transformer_codebook_sizes = (
            *codebook_sizes,
            collision_codebook_size,
        )
    else:
        semantic_df = raw_semantic_df
        transformer_codebook_sizes = codebook_sizes

    cfg = replace(
        base_transformer_cfg,
        rq_num_levels=rq_num_levels,
        num_levels=len(transformer_codebook_sizes),
        codebook_sizes=transformer_codebook_sizes,
        checkpoint_path=output_dir / "best_model.pt",
    )

    train_examples, val_examples, item2idx, _ = prepare_examples(
        data=data,
        semantic_df=semantic_df,
        cfg=cfg,
    )
    val_examples = limit_examples(
        val_examples,
        count=cfg.val_subset_size,
        seed=cfg.seed,
    )
    train_loader, val_loader = build_loaders(
        train_examples,
        val_examples,
        cfg,
    )

    sid_to_items = None
    if not resolve_collisions:
        sid_to_items = build_sid_to_items(
            semantic_df,
            item2idx,
        )

    model = SemanticTransformerModel(
        num_items=len(item2idx),
        cfg=cfg,
    ).to(cfg.device)

    training_history = train_semantic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        sid_to_items=sid_to_items,
        metric_logger=metric_logger,
    )

    return {
        "experiment_name": experiment_name,
        "method": method,
        "collision_mode": collision_mode,
        "resolve_collisions": resolve_collisions,
        "rq_num_levels": rq_num_levels,
        "transformer_num_levels": len(transformer_codebook_sizes),
        "codebook_sizes": codebook_sizes,
        "transformer_codebook_sizes": transformer_codebook_sizes,
        "sid_metrics": sid_metrics,
        "training_history": training_history,
    }


def run_all_experiments(data, embeddings, transformer_cfg):
    all_results = []

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for method in SID_Methods:
        for setup in SID_Hyperparametrs:
            codebook_sizes = resolve_codebook_sizes(
                codebook_sizes=setup.get("codebook_sizes"),
                num_levels=setup.get("num_levels"),
                codebook_size=setup.get("codebook_size"),
            )
            run_name = (
                f"{method}_C{'-'.join(map(str, codebook_sizes))}"
            )

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "method": method,
                    "codebook_sizes": ",".join(
                        map(str, codebook_sizes)
                    ),
                    "rq_num_levels": len(codebook_sizes),
                    "num_beams": transformer_cfg.num_beams,
                    "recall_k": transformer_cfg.recall_k,
                })

                def sid_metric_logger(metrics, step=None):
                    log_mlflow_metrics(
                        metrics,
                        step=step,
                        prefix="sid_training",
                    )

                raw_semantic_df, sid_metrics = build_raw_semantic_ids(
                    method=method,
                    embeddings=embeddings,
                    codebook_sizes=codebook_sizes,
                    metric_logger=sid_metric_logger,
                )
                log_mlflow_metrics(sid_metrics, prefix="sid")

                for resolve_collisions in COLLISION_RESOLUTION_MODES:
                    cfg = replace(
                        transformer_cfg,
                        resolve_collisions=resolve_collisions,
                    )
                    collision_mode = (
                        "with_collision_resolution"
                        if resolve_collisions
                        else "without_collision_resolution"
                    )

                    def metric_logger(metrics, step=None):
                        log_mlflow_metrics(
                            metrics,
                            step=step,
                            prefix=collision_mode,
                        )

                    result = run_single_experiment(
                        method=method,
                        codebook_sizes=codebook_sizes,
                        data=data,
                        raw_semantic_df=raw_semantic_df,
                        sid_metrics=sid_metrics,
                        base_transformer_cfg=cfg,
                        metric_logger=metric_logger,
                    )

                    mlflow.log_param(
                        f"{collision_mode}_codebook_sizes",
                        ",".join(
                            map(
                                str,
                                result["transformer_codebook_sizes"],
                            )
                        ),
                    )

                    recall_name = f"recall@{cfg.recall_k}"
                    if result["training_history"]:
                        best_epoch = max(
                            result["training_history"],
                            key=lambda row: row[recall_name],
                        )
                        log_mlflow_metrics(
                            {
                                f"best_{recall_name}": (
                                    best_epoch[recall_name]
                                ),
                                "best_epoch": best_epoch["epoch"],
                            },
                            prefix=collision_mode,
                        )

                    output_dir = (
                        Path("reports/experiments")
                        / result["experiment_name"]
                    )
                    results_path = output_dir / "results.json"

                    with results_path.open(
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(
                            result,
                            file,
                            ensure_ascii=False,
                            indent=2,
                        )

                    mlflow.log_artifact(
                        str(results_path),
                        artifact_path=collision_mode,
                    )
                    all_results.append(result)

    return all_results


def main():
    transformer_cfg = make_transformer_config()
    data, embeddings = load_yambda_data(
        transformer_cfg.count_of_samples
    )
    run_all_experiments(
        data=data,
        embeddings=embeddings,
        transformer_cfg=transformer_cfg,
    )
    print("Мы великолепны")


if __name__ == "__main__":
    main()