import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseScheduler(ABC, nn.Module):
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
    def var(self) -> torch.Tensor:
        return self._var

    @property
    def timesteps(self) -> int:
        return self._betas.size(0)

    def set_betas(self, betas: torch.Tensor) -> None:
        """
        Set betas and update alphas, bar_alphas, var

        :param betas:
        :type betas: torch.Tensor: (T,)
        """
        self._betas = betas
        self._alphas = 1.0 - betas
        self._bar_alphas = torch.cumprod(self._alphas, dim=0)
        self._var = (
            (1.0 - self.bar_alphas[:-1]) / (1.0 - self.bar_alphas[1:]) * self.betas
        )


class BaseNoiseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        :param x: noisy inputs
        :type x: torch.Tensor: (B, N)
        :param ts: timesteps for each object in batch
        :type ts: torch.Tensor: (B,)
        :return: predicted noise for each feature
        :rtype: torch.Tensor: (B, N)
        """
        pass
