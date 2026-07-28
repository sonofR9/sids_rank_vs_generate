import numpy as np
import polars as pl


def compute_pairwise_accuracy(
    uids: np.array,
    timestamps: np.array,
    labels: np.array,
    probs: np.array,
    session_gap_seconds: int = 15 * 60,):

    df = pl.DataFrame(
        {
            "uids": uids,
            "timestamp": timestamps,
            "labels": labels,
            "probs": probs,
        }
    )

    df = (
        df
        .with_row_index("row_id")
        .sort(["uids", "timestamp", "row_id"])
        .with_columns(
            next_time = pl.col("timestamp").shift(-1).over("uids"),
            next_lab = pl.col("labels").shift(-1).over("uids"),
            next_pro = pl.col("probs").shift(-1).over("uids"),
        )
        .filter(
            (pl.col("next_time") - pl.col("timestamp") <= session_gap_seconds) &
            (pl.col("labels") != pl.col("next_lab"))
        )
    )

    df = df.with_columns(
      score = pl
      .when((pl.col("next_pro") - pl.col("probs")) * (pl.col("next_lab") - pl.col("labels")) > 0)
      .then(1.0)
      .when((pl.col("next_pro") == pl.col("probs")))
      .then(0.5)
      .otherwise(0.0))

    accuracy = df.select(pl.col("score").mean()).item()
    if accuracy is None:
      return 0.0
    return accuracy