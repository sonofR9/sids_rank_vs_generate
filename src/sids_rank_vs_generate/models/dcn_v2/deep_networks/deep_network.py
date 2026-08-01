import torch
import torch.nn as nn

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units):
      super().__init__()
      dims = [input_dim] + hidden_units
      layers = []
      for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        layers.append(nn.ReLU())
      self.network = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.network(x)
