import typing

from . import base
from . import functional

if typing.TYPE_CHECKING:
    from . import diffusion


class CosineScheduler(base.Scheduler):
    def __init__(self, diffusion: "diffusion.Diffusion1d", s: float = 0.008):
        super().__init__()

        self.s_ = s
        self.set_betas(functional.cosine_beta_schedule(diffusion.timesteps_, s))


class LinearScheduler(base.Scheduler):
    def __init__(
        self,
        diffusion: "diffusion.Diffusion1d",
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ):
        super().__init__()

        self.beta_min_ = beta_min
        self.beta_max_ = beta_max
        self.set_betas(
            functional.linear_beta_schedule(diffusion.timesteps_, beta_min, beta_max)
        )
