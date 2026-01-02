import torch
from torch import nn
from . import base
from . import schedulers
from typing import Tuple

class DDPM1d(nn.Module):
    def __init__(self,
                 num_features: int,
                 timesteps: int = 1000,
                 noise_model: base.BaseNoiseModel = None,
                 scheduler: base.BaseScheduler | str = 'cosine'):
        super().__init__()

        self.num_features_ = num_features
        self.timesteps_ = timesteps
        self.noise_model_ = noise_model
        
        if isinstance(scheduler, base.BaseScheduler):
            assert scheduler.timesteps == timesteps, 'Schedule must be planned for each timestep'
            self.scheduler_ = scheduler
        elif isinstance(scheduler, str):
            if scheduler == 'cosine':
                self.scheduler_ = schedulers.CosineScheduler(timesteps)
            elif scheduler == 'linear':
                self.scheduler_ = schedulers.LinearScheduler(timesteps)
            else:
                raise ValueError
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to input using the forward diffusion process and predict the noise thereof
        
        :param x: Clean inputs x_0
        :type x: torch.Tensor: (B, N)
        :return: predicted and true noise for each feature
        :rtype: Tuple[torch.Tensor: (B, N), torch.Tensor: (B, N)]
        """
        ts = torch.randint(0, self.timesteps_, (x.size(0),))
        eps = torch.randn_like(x)
        bar_alphas = self.scheduler_.bar_alphas[ts].unsqueeze(1)
        diffusion = torch.sqrt(bar_alphas) * x + torch.sqrt(1.0 - bar_alphas) * eps
        return self.noise_model_(diffusion, ts), eps

    @torch.no_grad()
    def sample(self, n: int = 5):
        """
        Sample tabular rows using the reverse diffusion process
        
        :param n: sample size
        :type n: int
        """
        self.eval()
        self.noise_model_.eval()

        x = torch.randn(n, self.num_features_)

        for t in range(self.timesteps_ - 1, -1, -1):
            alpha_t = self.scheduler_.alphas[t]
            bar_alpha_t = self.scheduler_.bar_alphas[t]
            var_t = self.scheduler_.var[t]

            ts = torch.full((n,), t)
            hat_eps = self.noise_model_(x, ts)
            
            x = (x - (1.0 - alpha_t) / torch.sqrt(1.0 - bar_alpha_t) * hat_eps) / torch.sqrt(alpha_t)

            if t > 0:
                x += torch.sqrt(var_t) * torch.randn_like(x)

        return x

