import torch
import torch.nn as nn
from .deep_networks.deep_network import DeepNetwork
from .deep_networks.res_deep_network import ResDeepNetwork
from .deep_networks.dense_deep_network import DenseDeepNetwork
from .dcn_v2_encoders.categorical_encoder import CategoricalEncoder
from .dcn_v2_encoders.multivalent_encoder import MultivalentEncoder
from .dcn_v2_encoders.piecewise_linear_encoder import PiecewiseLinearEncoder
from .deep_networks.mixture_low_rank import MixtureLowRankCrossNetwork


def build_deep_network(
    input_dim: int,
    hidden_units: list[int],
    deep_type: str = "mlp",
) -> nn.Module:
    """Factory for the deep tower: ``mlp``, ``resnet``, or ``densenet``."""
    if deep_type == "mlp":
        return DeepNetwork(input_dim, hidden_units)
    if deep_type == "resnet":
        return ResDeepNetwork(input_dim, hidden_units)
    if deep_type == "densenet":
        return DenseDeepNetwork(input_dim, hidden_units)
    raise ValueError(f"Unknown deep_type={deep_type!r}, expected 'mlp', 'resnet', or 'densenet'")



class DCNV2(
    nn.Module
):  # it would be cool if authors would provide some base class for both dcnv2 and concat mlp
    # P.S. I am to lazy to refactor original class, so... ctr+c + ctrl+v mostly
    def __init__(
        self,
        embedding_size,
        cross_layers,
        deep_units,
        input_size,
        dense_train_df,
        n_bins,
        train_df_slice,
        cardinality=65536,
        num_experts: int = 4,
        low_rank: int = 32,
        deep_network: str = "mlp",
        output_size: int = 2,
        full_input_size_provided: bool = True,
    ):
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        super().__init__()

        self.embedding = nn.Embedding(cardinality, embedding_size)
        self.init_embedding_weights(self.embedding, init_range=0.02)
        self.categorical_encoder = CategoricalEncoder(self.embedding)
        self.multivalent_encoder = MultivalentEncoder(self.embedding)

        self.dense_encoder = PiecewiseLinearEncoder.from_dataset(
            dense_train_df,
            n_bins=n_bins,
            train_df_slice=train_df_slice,
        )

        total_input_size = (
            input_size + sum(self.dense_encoder.n_bins)
            if not full_input_size_provided
            else input_size
        )

        self.cross_network = MixtureLowRankCrossNetwork(
            input_dim=total_input_size,
            num_layers=cross_layers,
            num_experts=num_experts,
            rank=low_rank,
        )

        self.deep = build_deep_network(
            total_input_size, deep_units, deep_network
        )

        self.output = nn.Linear(deep_units[-1] + total_input_size, output_size)

    @staticmethod
    def init_embedding_weights(module: nn.Embedding, init_range: float = 0.02):
        nn.init.trunc_normal_(
            module.weight,
            std=init_range,
            a=-2 * init_range,
            b=2 * init_range,
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        #####################
        ### (づ•̀ᴗ•́)づ──☆*:・ﾟ
        #####################
        full_input = []
        full_input.append(self.dense_encoder(inputs["dense_features"]))

        for sparse_feature in ["uid", "item_id"]:
            full_input.append(
                self.categorical_encoder(
                    inputs["sparse_features"][sparse_feature]
                ).flatten(start_dim=1)
            )

        for multivalent_feature in ["artist_ids", "album_ids"]:
            full_input.append(
                self.multivalent_encoder(
                    inputs["multivalent_features"][multivalent_feature][
                        "values"
                    ],
                    inputs["multivalent_features"][multivalent_feature][
                        "lengths"
                    ],
                ).flatten(start_dim=1)
            )

        x = torch.cat(full_input, dim=1)

        cross_out = self.cross_network(x)
        deep_out = self.deep(x)

        combined = torch.cat([cross_out, deep_out], dim=1)
        output = self.output(combined)

        return output
