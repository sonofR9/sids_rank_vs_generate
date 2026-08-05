import torch
import torch.nn as nn
import polars as pl

class PiecewiseLinearEncoder(nn.Module):
    @staticmethod
    def compute_bins(
        X: torch.Tensor,
        n_bins: int,
    ) -> list[torch.Tensor]:
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        bin_edges = torch.quantile(
            X, torch.linspace(0, 1, n_bins + 1, device=X.device), dim=0
        )

        bins_per_feature = []
        for feature_idx in range(bin_edges.shape[1]):
            unique_edges = torch.unique(bin_edges[:, feature_idx])
            bins_per_feature.append(unique_edges)

        return bins_per_feature

    @classmethod
    def from_dataset(
        cls: "PiecewiseLinearEncoder",
        dense_train_df: pl.DataFrame,
        n_bins: int = 32,
        train_df_slice: int = 1_000_000,
    ) -> "PiecewiseLinearEncoder":
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        dense_trian_tensor = torch.tensor(
            dense_train_df.head(train_df_slice).to_numpy(), dtype=torch.float32
        )

        bins_per_feature = cls.compute_bins(dense_trian_tensor, n_bins)

        n_features = len(bins_per_feature)
        max_n_bins = max(len(b) - 1 for b in bins_per_feature)

        weight = torch.zeros(n_features, max_n_bins, dtype=torch.float32)
        bias = torch.zeros(n_features, max_n_bins, dtype=torch.float32)

        mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        single_bin_mask = torch.zeros(n_features, dtype=torch.bool)

        n_bins_list = []

        for feat_idx, edges in enumerate(bins_per_feature):
            n_bins = len(edges) - 1
            n_bins_list.append(n_bins)

            assert n_bins >= 1, f"Feature {feat_idx} has only one unique value."

            for bin_idx in range(n_bins):
                a, b = edges[bin_idx], edges[bin_idx + 1]
                width = (b - a).item()
                weight[feat_idx, bin_idx] = 1.0 / max(width, 1e-8)
                bias[feat_idx, bin_idx] = -a / max(width, 1e-8)

            mask[feat_idx, :n_bins] = True
            single_bin_mask[feat_idx] = n_bins == 1

        return cls(weight, bias, mask, n_bins_list, single_bin_mask)

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        n_bins: int,
        single_bin_mask,
    ):
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        super().__init__()

        self.weight = nn.Buffer(weight)
        self.bias = nn.Buffer(bias)
        self.single_bin_mask = nn.Buffer(single_bin_mask)
        self.mask = nn.Buffer(mask)
        self._n_bins = n_bins

    @property
    def n_bins(self):
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        return self._n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        encoded = self.bias + self.weight * x.unsqueeze(-1)

        if encoded.size(-1) >= 1:
            encoded[..., 0] = torch.clamp(encoded[..., 0], max=1.0)

        if encoded.size(-1) >= 2:
            if encoded.size(-1) > 2:
                encoded[..., 1:-1] = torch.clamp(
                    encoded[..., 1:-1], min=0.0, max=1.0
                )
            encoded[..., -1] = torch.clamp(encoded[..., -1], min=0.0)

        encoded[:, self.single_bin_mask, 0] = encoded[
            :, self.single_bin_mask, 0
        ].clamp(0.0, 1.0)

        flat_encoded = encoded.flatten(start_dim=1)

        flat_mask = self.mask.flatten()
        result = flat_encoded[:, flat_mask]

        return result
