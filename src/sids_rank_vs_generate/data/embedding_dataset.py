import numpy as np
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx]
