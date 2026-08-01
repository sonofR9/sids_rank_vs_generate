import polars as pl
import numpy as np
from collections import defaultdict
# давай все тесты в один запихнем
def tests_semantic_ids(df: pl.DataFrame, num_levels=3, codebook_size=512, item_col="item_id"):
    '''
    Здесь мы посчитаем independent code rate, codebook load balance
    Independent Code Rate (ICR) = n_unique_used_codes / n_items
    Codebook Load Balance:
    For each level l and code k:
    load[l, k] = count[l, k] / (n_items / codebook_size)
    '''
    sid_cols_names = [f"sid_{i}" for i in range(num_levels)]
    sid_cols = df.select(sid_cols_names)
    lists_for_sid_counter = [[0] * codebook_size for _ in range(num_levels)]
    unique_sid = defaultdict(int)
    for row in sid_cols.iter_rows():
        unique_sid[row]+=1
        for level, code in enumerate(row):
            lists_for_sid_counter[level][code]+=1
    n_items = df.height
    print(f'ICR = {len(unique_sid)/n_items}')
    for level in range(num_levels):
        counts = np.array(lists_for_sid_counter[level])
        loads = counts/(n_items/codebook_size)
        q1, q2, q3 = np.percentile(loads, [25, 50, 75])
        print(f'Первый квантиль по CLB = {q1}')
        print(f'Второй квантиль по CLB = {q2}')
        print(f'Третий квантиль по CLB = {q3}')
    print(f"max_collision: {max(unique_sid.values())}")
