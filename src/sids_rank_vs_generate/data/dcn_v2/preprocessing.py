import polars as pl
import torch

def join_item_artist_album(
    listens: pl.DataFrame,
    artists: pl.DataFrame,
    albums: pl.DataFrame,) -> pl.DataFrame:

    arts = artists.group_by("item_id").agg(
        pl.col("artist_id").alias("artist_ids")
    )

    alb = albums.group_by("item_id").agg(
        pl.col("album_id").alias("album_ids")
    )

    listens = listens.join(arts, on="item_id", how="left").join(alb, on="item_id", how="left")
    return listens


def temporal_train_test_split(
    df: pl.DataFrame,
    test_last_seconds: float,
    time_column: str = "timestamp",) -> tuple[pl.DataFrame, pl.DataFrame]:

    mx_time = df.select(pl.col("timestamp").max()).item()
    train = df.filter(pl.col("timestamp") < mx_time - test_last_seconds)
    test = df.filter(pl.col("timestamp") >= mx_time - test_last_seconds)

    return (train, test)


def to_device(obj, device: torch.device | str):
    if isinstance(obj, torch.Tensor):
      return obj.to(device)
    elif isinstance(obj, dict):
      return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
      return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
      return tuple(to_device(x, device) for x in obj)
    else:
      return obj

    
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
