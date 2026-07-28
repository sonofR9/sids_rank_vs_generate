import warnings
from src.sids_rank_vs_generate.data.dcn_v2.preprocessing import extract_preference_pairs
import gc
import polars as pl
from huggingface_hub import hf_hub_download
from experiments.dcn_v2.dataset_config import DatasetConfig

warnings.filterwarnings("ignore")
path = hf_hub_download(
    repo_id="matfu21/yambda-50m-lag-features",
    repo_type="dataset",
    filename="listens.parquet",
)
listens = pl.read_parquet(path)
listens = extract_preference_pairs(listens)
dataset_config = DatasetConfig()

if __name__ == "__main__":
    raise SystemExit(main())