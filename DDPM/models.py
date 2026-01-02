import torch
from torch import nn
from .base import IDDPM1dModel

class DDPM1dDefaultModel(IDDPM1dModel):
    def __init__(self,
                 num_features: int,
                 timesteps: int = 1000,
                 hidden: int = 256):
        super().__init__()
        self.num_features = num_features
        self.timesteps = timesteps
        self.hidden = hidden

        self.emb = nn.Embedding(timesteps + 1, hidden)

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        return super().forward(x, ts)
    