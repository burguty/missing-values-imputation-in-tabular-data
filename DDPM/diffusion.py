from typing import Tuple

import torch
from torch import nn

from . import base
from . import schedulers


class Diffusion1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        timesteps: int = 1000,
        denoiser: base.Denoiser1dModel = None,
        scheduler: base.Scheduler = None,
    ):
        super().__init__()

        self.num_features_ = num_features
        self.timesteps_ = timesteps

        if denoiser is None:
            self.denoiser_ = None
        elif isinstance(denoiser, base.Denoiser1dModel):
            self.denoiser_ = denoiser
        else:
            raise ValueError

        if scheduler is None:
            self.scheduler_ = schedulers.CosineScheduler(self)
        elif isinstance(scheduler, base.Scheduler):
            assert (
                scheduler.timesteps == timesteps
            ), "Schedule must be planned for each timestep"
            self.scheduler_ = scheduler
        else:
            raise ValueError

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imitates the forward diffusion process for a randomly sampled timestep

        Args:
            x (torch.Tensor): Clean inputs. Shape: (B, num_features)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - noisy inputs. Shape: (B, num_features)
                - timestep for each sample. Shape: (B,)
                - noise for each object. Shape: (B, num_features)
        """

        ts = torch.randint(0, self.timesteps_, (x.size(0),))
        noise = torch.randn_like(x)
        bar_alphas = self.scheduler_.bar_alphas[ts].unsqueeze(1)
        diffused = torch.sqrt(bar_alphas) * x + torch.sqrt(1.0 - bar_alphas) * noise
        return diffused, ts, noise

    @torch.no_grad()
    def sample(self, n: int = 5, variance_strategy: str = "posterior") -> torch.Tensor:
        """
        Sample tabular rows using the reverse diffusion process

        Args:
            n (int, optional): Sample size. Defaults to 5.

            variance_strategy (str, optional):
                Strategy to choose the reverse-process. \n
                Possible values: 'posterior', 'constant'. \n
                Defaults to 'posterior'.

        Returns:
            torch.Tensor: Generated samples. Shape: (n, num_features)
        """
        self.eval()

        if self.denoiser_ is not None:
            self.denoiser_.eval()

        x = torch.randn(n, self.num_features_)

        for t in range(self.timesteps_ - 1, -1, -1):
            beta_t = self.scheduler_.betas[t]
            alpha_t = self.scheduler_.alphas[t]
            bar_alpha_t = self.scheduler_.bar_alphas[t]

            ts = torch.full((n,), t)
            hat_eps = self.denoiser_(x, ts) if self.denoiser_ is not None else x

            x = (
                x - (1.0 - alpha_t) / torch.sqrt(1.0 - bar_alpha_t) * hat_eps
            ) / torch.sqrt(alpha_t)

            if t > 0:
                if variance_strategy == "posterior":
                    bar_alpha_t_1 = self.scheduler_.bar_alphas[t - 1]
                    var = beta_t * (1.0 - bar_alpha_t_1) / (1.0 - bar_alpha_t)
                elif variance_strategy == "constant":
                    var = beta_t
                else:
                    raise ValueError

                x += torch.sqrt(var) * torch.randn_like(x)

        return x
