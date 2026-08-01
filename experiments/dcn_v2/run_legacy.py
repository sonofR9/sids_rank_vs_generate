import warnings
import gc
from collections import defaultdict

import polars as pl
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from experiments.dcn_v2.dataset_config import DatasetConfig

from sids_rank_vs_generate.data.dcn_v2.preprocessing import (
    extract_preference_pairs,
    join_item_artist_album,
    temporal_train_test_split,
)
from sids_rank_vs_generate.data.dcn_v2.yambda_dataset import YambdaDataset
from sids_rank_vs_generate.evaluation.metrics.dcn_v2.compute_pairwise_accuracy import (
    compute_pairwise_accuracy,
)
from sids_rank_vs_generate.data.dcn_v2.ranker_dataset import RankerDataset
from sids_rank_vs_generate.data.dcn_v2.multihash_transform import MultihashTransform
from sids_rank_vs_generate.models.dcn_v2.dcn_v2 import DCNV2
from sids_rank_vs_generate.training.dcn_v2.train_and_eval import train_model

warnings.filterwarnings("ignore")

def main():
    path = hf_hub_download(
        repo_id="matfu21/yambda-50m-lag-features",
        repo_type="dataset",
        filename="listens.parquet",
    )
    listens = pl.read_parquet(path)
    listens = extract_preference_pairs(listens)
    dataset_config = DatasetConfig()
    yambda_dataset = YambdaDataset(
        dataset_type=dataset_config.dataset_type,
        dataset_size=dataset_config.dataset_size
    )
    albums = yambda_dataset.album_item_mapping().to_polars()
    artists = yambda_dataset.artist_item_mapping().to_polars()
    listens = join_item_artist_album(listens, artists, albums)
    del artists, albums
    gc.collect()
    train_listens, test_listens = temporal_train_test_split(listens, test_last_seconds=30 * 24 * 60 * 60)
    del listens
    gc.collect()
    test_listens = test_listens.with_columns(
        random_score=(
            pl.int_range(0, pl.len())
            .hash(seed=42)
            .cast(pl.Float64)
            / float(2**64 - 1)
        )
    )
    metrics = {
        "pair_accuracy_like": compute_pairwise_accuracy(test_listens.select("uid"), test_listens.select("timestamp"), test_listens.select("is_like").cast(pl.Int8), test_listens.select("random_score"),),
        "pair_accuracy_full_play": compute_pairwise_accuracy(test_listens.select("uid"), test_listens.select("timestamp"), test_listens.select("is_full_play").cast(pl.Int8), test_listens.select("random_score"),)
    }


    CARDINALITY = 2**22

    ITEM_ID_SEEDS = [42, 43, 44]
    UID_SEEDS = [45, 46]
    ATRIST_ID_SEEDS = [41, 40]
    ALBUM_IDS_SEEDS = [38, 37]
    multihash_transform = MultihashTransform(
        sparse_features_config={
            "item_id": ITEM_ID_SEEDS,
            "uid": UID_SEEDS,
        },
        sparse_features_name="sparse_features",
        multivalent_features_config={
            "artist_ids": ATRIST_ID_SEEDS,
            "album_ids": ALBUM_IDS_SEEDS,
        },
        multivalent_features_name="multivalent_features",
        cardinality=CARDINALITY,
    )
    TOTAL_SPARSE_FEATURES = (
        len(ITEM_ID_SEEDS)
        + len(UID_SEEDS)
        + len(ATRIST_ID_SEEDS)
        + len(ALBUM_IDS_SEEDS)
    )

    train_dataset = RankerDataset(
        train_listens,
        [multihash_transform],
        label_columns=list(dataset_config.LABEL_COLUMNS),
        dense_columns=list(dataset_config.DENSE_COLUMNS),
        sparse_columns=list(dataset_config.SPARSE_COLUMNS),
        multivalent_columns=list(dataset_config.MULTIVALENT_COLUMNS),
        batch_size=512,
    )

    test_dataset = RankerDataset(
        test_listens,
        [multihash_transform],
        label_columns=list(dataset_config.LABEL_COLUMNS),
        dense_columns=list(dataset_config.DENSE_COLUMNS),
        sparse_columns=list(dataset_config.SPARSE_COLUMNS),
        multivalent_columns=list(dataset_config.MULTIVALENT_COLUMNS),
        batch_size=1024,
    )

    train_listens_dense_million = train_listens[dataset_config.DENSE_COLUMNS].slice(0, 1_000_000)
    gc.collect()


    common_params = dict(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: batch[0],
    )

    train_len = len(train_dataset)
    train_loader_small = DataLoader(
        torch.utils.data.Subset(
            train_dataset, range(train_len * 3 // 4, train_len)
        ),
        shuffle=True,
        **common_params,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_params,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_params,
    )
    embedding_dim = 16

    metrics_dcnv2_mlp = defaultdict(float)
    n_runs = 3
    for i in range(n_runs):
        model_dcnv2_mlp = DCNV2(
        
            embedding_size=embedding_dim,
            cross_layers=3,
            deep_units=[256, 128, 64, 32],
            input_size=embedding_dim * TOTAL_SPARSE_FEATURES,
            dense_train_df=train_listens_dense_million,
            n_bins=32,
            train_df_slice=1_000_000,
            cardinality=CARDINALITY,
            num_experts=4,
            low_rank=32,
            deep_network="mlp",
            output_size=1,
            full_input_size_provided=False,
        )

        metrics = train_model(
            #####################
            ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
            #####################
            model_dcnv2_mlp,
            train_loader=train_loader,
            # train_loader=train_loader_small,
            test_loader=test_loader,
            epochs=1,
            lr=1e-3,
            embed_lr=3e-4,
            train_log_every=50,
        )
        for key, value in metrics.items():
            metrics_dcnv2_mlp[key] += value

    for key, value in metrics_dcnv2_mlp.items(): # возможно ошибка, должно быть metrics_concat_mlp
        metrics_dcnv2_mlp[key] /= n_runs 
    print(f"averaged {metrics_dcnv2_mlp}")


if __name__ == "__main__":
    raise SystemExit(main())
