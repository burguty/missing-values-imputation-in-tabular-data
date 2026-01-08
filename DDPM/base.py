import torch
from torch import nn
from abc import ABC, abstractmethod


class DDPMScheduler(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def betas(self) -> torch.Tensor:
        return self._betas

    @property
    def alphas(self) -> torch.Tensor:
        return self._alphas

    @property
    def bar_alphas(self) -> torch.Tensor:
        return self._bar_alphas

    @property
    def timesteps(self) -> int:
        return self._betas.size(0)

    def set_betas(self, betas: torch.Tensor) -> None:
        """Set betas. Update alphas, bar_alphas

        Args:
            betas (torch.Tensor): Shape: (T,)
        """
        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas", 1.0 - betas)
        self.register_buffer("_bar_alphas", torch.cumprod(self._alphas, dim=0))


class Denoiser1dModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Trying to predict added noise for each object.

        Args:
            x (torch.Tensor): Noisy inputs. Shape: (B, num_features)
            ts (torch.Tensor): Timesteps for each object in batch. Shape: (B,)

        Returns:
            torch.Tensor: predicted noise for each object. Shape: (B, num_features)
        """
        pass
