import torch
from torch import nn
from .base import Denoiser1dModel


class Diffusion1dDefaultModel(Denoiser1dModel):
    def __init__(self, num_features: int, timesteps: int = 1000, hidden: int = 128):
        super().__init__()
        self.num_features = num_features
        self.timesteps = timesteps
        self.hidden = hidden

        self.emb = nn.Embedding(timesteps, hidden)
        self.net = nn.Sequential(
            nn.Linear(num_features + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_features),
        )

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        ts_emb = self.emb(ts)
        x_ts_emb = torch.concat([x, ts_emb], dim=1)
        return self.net(x_ts_emb)
