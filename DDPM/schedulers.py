from .base import DDPMScheduler
from . import functional


class CosineScheduler(DDPMScheduler):
    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        super().__init__()
        self.s_ = s
        self.set_betas(functional.cosine_beta_schedule(timesteps, s))


class LinearScheduler(DDPMScheduler):
    def __init__(
        self, timesteps: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02
    ):
        super().__init__()
        self.beta_min_ = beta_min
        self.beta_max_ = beta_max
        self.set_betas(functional.linear_beta_schedule(timesteps, beta_min, beta_max))
