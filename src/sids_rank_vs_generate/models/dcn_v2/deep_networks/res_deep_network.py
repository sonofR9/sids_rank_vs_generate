import torch
import torch.nn as nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.first_l = nn.Linear(in_dim, out_dim)
        self.second_l = nn.Linear(out_dim, out_dim)

        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            skip = x
        else:
            skip = self.proj(x)

        out = self.first_l(x)
        relu = nn.ReLU()
        out = relu(out)
        out = self.second_l(out)

        out = out + skip
        return relu(out)

class ResDeepNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_units: list[int]):
        super().__init__()
        dims = [input_dim] + hidden_units
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(ResidualMLPBlock(dims[i], dims[i + 1]))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x
