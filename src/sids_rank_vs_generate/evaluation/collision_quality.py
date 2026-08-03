import numpy as np
import polars as pl

from sids_rank_vs_generate.sids.collision_resolution import get_sid_cols

'''
Path Average Similarity (PAS) = mean embedding similarity для айтемов внутри коллизии. Считается по путям (из GPR)
'''
def PAS(sids: pl.DataFrame, embeddings: pl.DataFrame) -> float:
    sid_cols = get_sid_cols(sids)
    collisions = (sids.join(
            embeddings,
            on="item_id",
            how="inner",
        )
        .group_by(sid_cols)
        .agg(
            pl.col("normalized_embed").alias("embeddings"),
            pl.len().alias("collision_size"),
        )
        .filter(pl.col("collision_size") > 1)
    )

    pas_values = []
    '''
    можно было честно перемножить скалярно, но, надеюсь, матрицами быстрее (мб ошибаюсь)
    '''
    for row in collisions.iter_rows(named=True):
        embeds = np.asarray(row["embeddings"], dtype=np.float32)
        n = len(embeds)
        matrix = embeds @ embeds.T
        pas_values.append(
            (matrix.sum() - np.trace(matrix)) / (n * (n - 1))
        )
    return float(np.mean(pas_values)) if pas_values else float("nan") # коллизий может и не быть, тогда нан 


'''
Codebook Load Balance (P25, P75, P90) = перцентили отклонений размеров кластеров от случая равномерного распределения 
айтемов по кодбуку (n_items / codebook_size). Считается по кодбукам и усредняется. Хотя вообще говоря можно считать 
на каждый кодбук по отдельности для полной картины. 
Не должна иметь слишком сильные перекосы от 100% на каждом перцентиле (из OxygenREC)
'''
def codebook_load_balance_and_coverage_and_utitlization(sids: pl.DataFrame, codebook_sizes: list[int]):
    sid_cols = get_sid_cols(sids)
    n = sids.height
    metrics = {}
    load_balance_values = {"p25": [], "p75": [], "p90": []}
    n_possible_paths = 1

    for layer, (col, size) in enumerate(zip(sid_cols, codebook_sizes), start=1):
        theoretical_mean = n / size
        counts = sids.group_by(col).agg(pl.len().alias("count")).get_column("count")
        counts = pl.concat([counts, pl.Series("count", [0] * (size - len(counts)))])

        p25 = counts.quantile(0.25, interpolation="higher") / theoretical_mean
        p75 = counts.quantile(0.75, interpolation="higher") / theoretical_mean
        p90 = counts.quantile(0.90, interpolation="higher") / theoretical_mean

        metrics[f"load_balance_p25_at_{layer}_layer"] = float(p25)
        metrics[f"load_balance_p75_at_{layer}_layer"] = float(p75)
        metrics[f"load_balance_p90_at_{layer}_layer"] = float(p90)

        load_balance_values["p25"].append(p25)
        load_balance_values["p75"].append(p75)
        load_balance_values["p90"].append(p90)

        n_possible_paths *= size
        n_unique_paths = sids.select(sid_cols[:layer]).unique().height
        metrics[f"codebook_coverage_at_{layer}_layer"] = n_unique_paths / n_possible_paths

        n_used = sids.get_column(col).n_unique()
        metrics[f"codebook_utilization_rate_at_{layer}_layer"] = n_used / size


    metrics["load_balance_p25_mean"] = float(np.mean(load_balance_values["p25"]))
    metrics["load_balance_p75_mean"] = float(np.mean(load_balance_values["p75"]))
    metrics["load_balance_p90_mean"] = float(np.mean(load_balance_values["p90"]))

    return metrics
