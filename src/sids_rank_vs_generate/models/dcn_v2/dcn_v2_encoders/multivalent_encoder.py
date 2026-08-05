import torch
import torch.nn as nn
import torch.nn.functional as F

class MultivalentEncoder(nn.Module):
    def __init__(self, embeddings: nn.Embedding):
        super().__init__()
        self.emb = embeddings

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        num_hashes = ids.shape[1]
        ans = []

        for i in range(num_hashes):
            now = ids[:, i]

            off = torch.cumsum(lengths, 0)
            off = torch.cat(
                (
                    torch.tensor([0], device=lengths.device),
                    off
                )
            )
            off = off[:-1]

            res = F.embedding_bag(
                now,
                self.emb.weight,
                offsets=off
            )

            ans.append(res)

        stacked = torch.stack(ans)
        return stacked.permute(1, 0, 2)
