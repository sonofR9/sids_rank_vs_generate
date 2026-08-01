import kagglehub
import polars as pl
from kagglehub import KaggleDatasetAdapter


dataset_path = "thekabeton/ysda-recsys-2026-yambda-dataset"

# count_of_samples - сколько берем последних взаимодействий
def load_yambda_data(count_of_samples: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    data = (
        kagglehub.dataset_load(
            KaggleDatasetAdapter.POLARS,  # в каком виде хочу увидеть данные (варианты: pandas, polars, numpy, pytorch...)
            dataset_path,
            "likes.parquet")
        .collect()
        .sort("timestamp", descending=True)
        [:count_of_samples])
    
    embeddings = (
        kagglehub.dataset_load(
            KaggleDatasetAdapter.POLARS,
            dataset_path,
            "embeddings_small.parquet")
        .collect())
    # Оставляем эмбеды только для items из interactions.
    item_ids = data.select("item_id").unique().get_column("item_id")

    embeddings = embeddings.filter(
        pl.col('item_id').is_in(item_ids.implode()) #
    )['item_id', 'normalized_embed']

    # Удаляем interactions без эмбедов
    items_with_embeddings = embeddings["item_id"].unique()

    data = data.filter(pl.col("item_id").is_in(items_with_embeddings.implode()))

    return data, embeddings
