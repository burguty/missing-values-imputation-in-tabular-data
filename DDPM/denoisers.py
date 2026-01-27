import torch
from torch import nn

from . import base
from . import modules


class SimpleMLP(base.Denoiser1dModel):
    def __init__(
        self, num_features: int, timestep_embed_dim: int = 20, hidden: int = 128
    ):
        super().__init__()

        self.num_features_ = num_features
        self.timestep_embed_dim_ = timestep_embed_dim
        self.hidden_ = hidden

        self.timestep_embed = nn.Sequential(
            modules.SinusoidalEmbedding(timestep_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(timestep_embed_dim, timestep_embed_dim),
        )

        self.fc1 = nn.Linear(num_features + timestep_embed_dim, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, num_features)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor):
        timesteps_emb = self.timestep_embed(timesteps)
        out = torch.cat([x, timesteps_emb], dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
