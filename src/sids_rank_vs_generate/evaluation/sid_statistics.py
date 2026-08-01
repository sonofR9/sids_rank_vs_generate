import polars as pl
import numpy as np
from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols

# давай все тесты в один запихнем
def tests_semantic_ids(df: pl.DataFrame, num_levels=3, codebook_size=512, item_col="item_id"):
    '''
    Здесь мы посчитаем independent code rate, codebook load balance
    Independent Code Rate (ICR) = n_unique_used_codes / n_items
    n_unique_codes = количество кодов, нашедших маппинг в айтемы (n_unique_used_codes по сути, оно же количество актуальных путей)
    Collisions rate = (n_items - n_unique_used_codes) / n_items = 1 - ICR
    SIDs uniqueness = то ли доля айтемов каталога, получивших уникальный SID, то ли доля семантиков, которые мапятся в уникальный айтем (из plum) != ICR
    я про юникнес понял что это типа Singleton Rate
    SID Collision percentiles (P90, P99, P999) = количество айтемов в коллизиях на перцентилях. Считается по путям (из OxygenREC)
    '''
    sid_cols_names = get_sid_cols(df)
    path_stats = df.group_by(sid_cols_names).len().rename({"len": "collision_size"})

    n_items = df.height
    n_unique_codes = path_stats.height
    n_singletons = path_stats.filter(pl.col("collision_size") == 1).height
    collision_stats = path_stats.filter(pl.col("collision_size") > 1)

    p90, p99, p999 = collision_stats.select(
        pl.col("collision_size").quantile(0.9, interpolation="higher").fill_null(0),
        pl.col("collision_size").quantile(0.99, interpolation="higher").fill_null(0),
        pl.col("collision_size").quantile(0.999, interpolation="higher").fill_null(0),
    ).row(0)

    return {
        "ICR": n_unique_codes / n_items,
        "n_unique_codes": n_unique_codes,
        "max_collision": path_stats.get_column("collision_size").max(),
        "Collisions rate": 1 - n_unique_codes / n_items,
        "n_singletons": n_singletons,
        "SIDs uniqueness": n_singletons / n_items,
        "SID Collision P90": int(p90),
        "SID Collision P99": int(p99),
        "SID Collision P999": int(p999),
    }