import polars as pl
import numpy as np
from collections import defaultdict
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
    sid_cols_names = [f"sid_{i}" for i in range(num_levels)]
    sid_cols = df.select(sid_cols_names)
    unique_sid = defaultdict(int)
    for row in sid_cols.iter_rows():
        unique_sid[row]+=1

    n_items = df.height
    n_singletons = sum(1 for x in unique_sid.values() if x==1)
    collision_sizes = np.array(list(unique_sid.values()))
    p90, p99, p999 = np.percentile(
        collision_sizes,
        [90, 99, 99.9],
        method="higher",
    )
    return {'ICR':len(unique_sid)/n_items,
    'n_unique_codes':len(unique_sid),
    'max_collision':max(unique_sid.values()),
    'Collisions rate':1-len(unique_sid)/n_items,
    "n_singletons": int(n_singletons),
    "SIDs uniqueness": n_singletons / n_items,
    "SID Collision P90": int(p90),
    "SID Collision P99": int(p99),
    "SID Collision P999": int(p999),
    }