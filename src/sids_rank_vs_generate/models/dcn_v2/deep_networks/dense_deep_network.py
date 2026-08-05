import torch
import torch.nn as nn

class DenseDeepNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()

        in_dim = input_dim
        for out_dim in hidden_units:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim += out_dim

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]

        for layer in self.layers:
            inp = torch.cat(features, dim=1)
            out = self.relu(layer(inp))
            features.append(out)

        return features[-1]
