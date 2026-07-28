import torch
import polars as pl
from typing import Any, Callable
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from collections import defaultdict

class RankerDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        transforms: list[Callable[[Any], Any]],
        label_columns: list[str],
        dense_columns: list[str],
        sparse_columns: list[str],
        multivalent_columns: list[str],
        batch_size: int,):

        if batch_size < 1:
          raise ValueError("batch должен быть хотя бы 1")

        self.batches = []
        num_rows = len(df)

        for start_idx in tqdm(range(0, num_rows, batch_size)):
            end_idx = min(start_idx + batch_size, num_rows)
            batch_df = df[start_idx:end_idx]

            batch_dict = defaultdict(dict)

            labels = batch_dict["labels"]
            for label_column in label_columns:
                labels[label_column] = torch.tensor(
                    batch_df[label_column].to_numpy(), dtype=torch.float32
                )

            batch_dict["dense_features"] = torch.tensor(
                batch_df.select(dense_columns).to_numpy(), dtype=torch.float32
            )

            sparse_features = batch_dict["sparse_features"]
            for sparse_column in sparse_columns:
                sparse_features[sparse_column] = torch.tensor(
                    batch_df[sparse_column].to_numpy(), dtype=torch.long
                )

            multivalent_features = batch_dict["multivalent_features"]
            for multivalent_column in multivalent_columns:
                values_list = []
                lengths_list = []

                for series in batch_df[multivalent_column]:
                    if series is None or len(series) == 0:
                        values_list.append(0)
                        lengths_list.append(0)
                    else:
                        values_list.extend(series)
                        lengths_list.append(len(series))

                multivalent_features[multivalent_column] = {
                    "values": torch.tensor(values_list, dtype=torch.long),
                    "lengths": torch.tensor(lengths_list, dtype=torch.long),
                }

            batch_dict["meta"] = {}
            for column in ["timestamp", "uid", "item_id"]:
                batch_dict["meta"][column] = torch.tensor(
                    batch_df[column].to_numpy(), dtype=torch.long
                )

            for transform in transforms:
                batch_dict = transform(batch_dict)

            self.batches.append(batch_dict)


    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.batches[idx]