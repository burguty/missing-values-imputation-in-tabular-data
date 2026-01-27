import torch
from torch import nn

from . import functional


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 20, moments: int = 5, std: float = 1.0):
        super().__init__()

        self.embedding_dim_ = embedding_dim
        self.moments_ = moments
        self.std_ = std

        self.c_ = nn.Parameter(torch.empty(moments))
        nn.init.normal_(self.c_, mean=0, std=std)
        self.fc = nn.Linear(2 * moments, embedding_dim)

    def forward(self, x: torch.Tensor):
        out = functional.sinusoidal_activation(x, self.c_)
        out = self.fc(out)
        return out
