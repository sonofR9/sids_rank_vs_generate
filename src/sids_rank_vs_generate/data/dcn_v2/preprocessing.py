import polars as pl

def extract_preference_pairs(df: pl.DataFrame) -> pl.DataFrame:
  return (
      df
      .with_row_index("row_id")
      .sort(["uid", "timestamp", "row_id"])
      .filter(
          (pl.col("is_like") != pl.col("is_like").shift(1).over("uid")) |
          (pl.col("is_like") != pl.col("is_like").shift(-1).over("uid")) |
          (pl.col("is_full_play") != pl.col("is_full_play").shift(1).over("uid")) |
          (pl.col("is_full_play") != pl.col("is_full_play").shift(-1).over("uid"))
      )
      .drop("row_id")
  )
