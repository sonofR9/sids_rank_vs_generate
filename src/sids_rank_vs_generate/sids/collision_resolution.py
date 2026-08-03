import polars as pl

def get_sid_cols(semantic_df):
    return sorted(
        [c for c in semantic_df.columns if c.startswith("sid_")],
        key=lambda x: int(x.split("_")[1]),
    )

def add_collision_resolving_sid(
    semantic_df: pl.DataFrame,
    item_col: str = "item_id",
    new_col: str | None = None,
):
    sid_cols = get_sid_cols(semantic_df)

    if new_col is None:
        last_col = sid_cols[-1]
        new_col = 'sid_'+str(int(last_col.split('_')[1])+1)
    result = semantic_df.with_columns(
        (pl.col(item_col)
        .rank('ordinal')
        .over(sid_cols)-1) # сдвигаю нумерацию
        .alias(new_col)
    )

    max_token = int(result[new_col].max())
    return result, max_token
