from typing import Any
import mmh3
import torch
import numpy as np

class MultihashTransform:
    def __init__(
        self,
        sparse_features_config: dict,
        sparse_features_name: str,
        multivalent_features_config: dict,
        multivalent_features_name: str,
        cardinality: int,
    ):
        self.sparse_features_config = sparse_features_config
        self.sparse_features_name = sparse_features_name
        self.multivalent_features_config = multivalent_features_config
        self.multivalent_features_name = multivalent_features_name
        self.cardinality = cardinality


    def _apply_multihash(
        self, tensor: torch.Tensor, seeds: list
    ) -> torch.Tensor:
        assert tensor.device.type == "cpu"
        tensor_np = tensor.numpy()

        hashed_tensors = []
        for seed in seeds:
            hashed_np = np.array(
                [
                    mmh3.hash(str(int(x)), seed) % self.cardinality
                    for x in tensor_np.flatten()
                ]
            ).reshape(tensor_np.shape)
            hashed_tensors.append(torch.from_numpy(hashed_np))

        return torch.stack(hashed_tensors, dim=-1).to(tensor.device)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        sparse_features = sample[self.sparse_features_name]
        for feature_name, seeds in self.sparse_features_config.items():
            sparse_features[feature_name] = self._apply_multihash(
                sparse_features[feature_name], seeds
            )

        multivalent_features = sample[self.multivalent_features_name]
        for feature_name, seeds in self.multivalent_features_config.items():
            multivalent_features[feature_name]["values"] = (
                self._apply_multihash(
                    multivalent_features[feature_name]["values"], seeds
                )
            )

        return sample