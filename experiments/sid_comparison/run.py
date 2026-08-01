import json
from dataclasses import replace
from pathlib import Path

from experiments.sid_comparison.config import (
    SID_Hyperparametrs,
    SID_Methods,
    make_rqvae_config,
    make_transformer_config,
    make_rqkmeans_config
)
from sids_rank_vs_generate.data.sem_trans_dataset import (
    build_loaders,
    limit_examples,
    prepare_examples,
)
from sids_rank_vs_generate.datasets_for_sem_transform import load_yambda_data
from sids_rank_vs_generate.evaluation.sid_statistics import (
    tests_semantic_ids,
)
from sids_rank_vs_generate.models.semantic_transformer_model import (
    SemanticTransformerModel,
)
from sids_rank_vs_generate.sids.collision_resolution import (
    add_collision_resolving_sid,
    get_sid_cols,
)
from sids_rank_vs_generate.sids.sid_builder import (
    build_semantic_ids,
)
from sids_rank_vs_generate.training.semantic_trainer import (
    train_semantic,
)


def build_sid_to_items(
    semantic_df,
    item2idx,
):
    """
    Строит:

        полный SID -> список внутренних item_idx

    Внутренние item_idx нужны потому, что batch["target"]
    в датасете содержит не исходный item_id, а item2idx[item_id].
    """

    sid_cols = get_sid_cols(semantic_df)
    sid_to_items = {}

    columns = [*sid_cols, "item_id"]

    for row in semantic_df.select(columns).iter_rows(named=True):
        sid = tuple(row[column] for column in sid_cols)
        item_id = row["item_id"]
        item_idx = item2idx[item_id]
        sid_to_items.setdefault(sid, []).append(item_idx)
    return sid_to_items


def run_single_experiment(
    method,
    rq_num_levels,
    codebook_size,
    data,
    embeddings,
    base_transformer_cfg,
):
    '''
    Основная функция, которая запускает эксперимент по обучение 
    трансформера на семантиках.
    1) Строит семантики (надо с конфигами разобраться!)
    2) Считаем статистики полученных семантиков (надо будет еще добавить)
    3) Выбираем ращрешаем ли коллизии
    4) Собираем датасет
    5) Обучаем трансформер
    6) Считаем метрики трансформера
    Мы великолепны
    '''
    if method == "rqkmeans":
        rqkmeans_config = make_rqkmeans_config(
            num_levels=rq_num_levels,
            codebook_size=codebook_size,
        )
        rqvae_config = None
    elif method == "rqvae":
        rqvae_config = make_rqvae_config(
            num_levels=rq_num_levels,
            codebook_size=codebook_size,
        )
        rqkmeans_config = None
    else:
        raise ValueError(f"Неизвестный метод построения семантиков: {method}")

    resolve_collisions = base_transformer_cfg.resolve_collisions

    collision_mode = (
        "without_collision"
        if resolve_collisions
        else "with_collision"
    )
    
    experiment_name = (
        f"{method}_L{rq_num_levels}_C{codebook_size}"
        f"_{collision_mode}"
    )

    print(f"\nStarting {experiment_name}")

    output_dir = (
        Path("reports/experiments")
        / experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Строим исходные Semantic IDs.
    raw_semantic_df = build_semantic_ids(
        method=method,
        embeddings=embeddings,
        rqvae_config=rqvae_config,
        rqkmeans_config=rqkmeans_config
    )

    # 2. Считаем статистики
    raw_sid_metrics = tests_semantic_ids(
        raw_semantic_df,
        num_levels=rq_num_levels,
    )

    # 3. Выбираем режим работы с коллизиями, если работаем без коллизий считаем новый набор статистик
    if resolve_collisions:
        semantic_df = add_collision_resolving_sid(
            semantic_df=raw_semantic_df,
            codebook_size=codebook_size,
        )
        # добавляем слой resolution токена 
        transformer_num_levels = rq_num_levels + 1

        resolved_sid_metrics = tests_semantic_ids(
            semantic_df,
            num_levels=transformer_num_levels,
        )

    else:
        semantic_df = raw_semantic_df
        transformer_num_levels = rq_num_levels
        resolved_sid_metrics = None

    # 4. Создаём отдельный конфиг для конкретного эксперимента
    cfg = replace(
        base_transformer_cfg,
        rq_num_levels=rq_num_levels,
        num_levels=transformer_num_levels,
        codebook_size=codebook_size,
    )

    # 5. Собираем дату
    train_examples, val_examples, item2idx, _ = (
        prepare_examples(
            data=data,
            semantic_df=semantic_df,
            cfg=cfg,
        )
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
    '''
    6. Для режима без resolution токена строим:
    
    SID -> все реальные item_idx с этим SID.
    
    В режиме с resolution-токеном каждый SID уже уникален,
    поэтому маппинг не требуется
    '''
    if resolve_collisions:
        sid_to_items = None
    else:
        sid_to_items = build_sid_to_items(
            semantic_df=semantic_df,
            item2idx=item2idx,
        )

    # 7. Создаём Semantic Transformer.
    model = SemanticTransformerModel(
        num_items=len(item2idx),
        cfg=cfg,
    ).to(cfg.device)
    '''
    8. Обучаем модель.
        
    resolve_collisions=True:
        вызвать валидацию по уникальным SID;
    
    resolve_collisions=False:
        раскрыть каждый beam SID через sid_to_items
        и считать item recall.
    '''
    training_history = train_semantic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        sid_to_items=sid_to_items,
    )

    result = {
        "experiment_name": experiment_name,
        "method": method,
        "resolve_collisions": resolve_collisions,
        "rq_num_levels": rq_num_levels,
        "transformer_num_levels": transformer_num_levels,
        "codebook_size": codebook_size,
        "raw_sid_metrics": raw_sid_metrics,
        "resolved_sid_metrics": resolved_sid_metrics,
        "training_history": training_history,
    }

    return result


def run_all_experiments(
    data,
    embeddings,
    transformer_cfg,
):
    all_results = []

    for method in SID_Methods:
        for setup in SID_Hyperparametrs:
            result = run_single_experiment(
                method=method,
                rq_num_levels=setup["num_levels"],
                codebook_size=setup["codebook_size"],
                data=data,
                embeddings=embeddings,
                base_transformer_cfg=transformer_cfg,
            )

            output_dir = (
                Path("reports/experiments")
                / result["experiment_name"]
            )

            with (
                output_dir / "results.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    result,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            all_results.append(result)

    return all_results

def main():
    transformer_cfg = make_transformer_config()

    data, embeddings = load_yambda_data(
        transformer_cfg.count_of_samples
    )

    results = run_all_experiments(
        data=data,
        embeddings=embeddings,
        transformer_cfg=transformer_cfg,
    )

    print("Мы великолепны")

if __name__ == "__main__":
    main()