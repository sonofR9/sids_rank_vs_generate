from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

class ExplicitFeedbackDataset(Dataset):
    def __init__(self, df):
        self.records = df.to_dict(
            orient="records"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def flatten_batch_field(batch, field):
    return [
        value
        for example in batch
        for value in example[field]
    ]


def collate_rich_varlen(batch):
    history_lengths = torch.tensor(
        [
            example["history_length"]
            for example in batch
        ],
        dtype=torch.long,
    )

    result = {
        "user_idx": torch.tensor(
            [x["user_idx"] for x in batch],
            dtype=torch.long,
        ),

        "candidate_item_idx": torch.tensor(
            [
                x["candidate_item_idx"]
                for x in batch
            ],
            dtype=torch.long,
        ),

        "candidate_timestamp": torch.tensor(
            [
                x["candidate_timestamp"]
                for x in batch
            ],
            dtype=torch.long,
        ),

        "candidate_is_organic": torch.tensor(
            [
                x["candidate_is_organic"]
                for x in batch
            ],
            dtype=torch.float32,
        ),

        "candidate_track_length_z": torch.tensor(
            [
                x["candidate_track_length_z"]
                for x in batch
            ],
            dtype=torch.float32,
        ),

        "candidate_has_track_length": torch.tensor(
            [
                x["candidate_has_track_length"]
                for x in batch
            ],
            dtype=torch.float32,
        ),

        "history_item_idxs_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_item_idxs",
            ),
            dtype=torch.long,
        ),

        "history_event_type_idxs_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_event_type_idxs",
            ),
            dtype=torch.long,
        ),

        "history_is_organic_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_is_organic",
            ),
            dtype=torch.float32,
        ),

        "history_played_ratio_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_played_ratio",
            ),
            dtype=torch.float32,
        ),

        "history_has_played_ratio_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_has_played_ratio",
            ),
            dtype=torch.float32,
        ),

        "history_track_length_z_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_track_length_z",
            ),
            dtype=torch.float32,
        ),

        "history_has_track_length_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_has_track_length",
            ),
            dtype=torch.float32,
        ),

        "history_time_delta_z_flat": torch.tensor(
            flatten_batch_field(
                batch,
                "history_time_delta_z",
            ),
            dtype=torch.float32,
        ),

        "history_lengths": history_lengths,

        "target": torch.tensor(
            [x["target"] for x in batch],
            dtype=torch.float32,
        ),
    }

    total_tokens = int(
        history_lengths.sum().item()
    )

    flat_fields = [
        "history_item_idxs_flat",
        "history_event_type_idxs_flat",
        "history_is_organic_flat",
        "history_played_ratio_flat",
        "history_has_played_ratio_flat",
        "history_track_length_z_flat",
        "history_has_track_length_flat",
        "history_time_delta_z_flat",
    ]

    for field in flat_fields:
        assert result[field].numel() == total_tokens

    return result

def make_dataloaders(
    train_encoded,
    val_encoded,
    test_encoded,
    batch_size=256,
    num_workers=0,
):
    train_dataset = ExplicitFeedbackDataset(train_encoded)
    val_dataset = ExplicitFeedbackDataset(val_encoded)
    test_dataset = ExplicitFeedbackDataset(test_encoded)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_rich_varlen,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader

