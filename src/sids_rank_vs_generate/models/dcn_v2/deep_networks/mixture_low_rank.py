import torch
import torch.nn as nn

class MixtureLowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, rank: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.rank = rank
        self.U = nn.Parameter(torch.randn(num_experts, input_dim, rank))
        self.V = nn.Parameter(torch.rand(num_experts, input_dim, rank))
        self.bias = nn.Parameter(torch.rand(input_dim))
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        expert_outputs = []
        for i in range(self.num_experts):
            y = xl @ self.V[i]
            z = y @ self.U[i].T
            z = z + self.bias
            z = x0 * z
            expert_outputs.append(z)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        weights = torch.softmax(self.gate(xl), dim=1).unsqueeze(-1)
        mixed = (expert_outputs * weights).sum(dim=1)
        return mixed + xl


class MixtureLowRankCrossNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        num_experts: int,
        rank: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MixtureLowRankCrossLayer(
                input_dim=input_dim,
                num_experts=num_experts,
                rank=rank,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x

        for layer in self.layers:
            xl = layer(x0, xl)
        return xl
