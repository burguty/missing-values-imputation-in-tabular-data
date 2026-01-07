import torch


def cosine_beta_schedule(timesteps, s: float = 0.008):
    t = torch.linspace(0, timesteps, timesteps + 1)
    f_t = torch.pow(torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5), 2)
    alpha_t = f_t / f_t[0]
    return torch.clip(1.0 - alpha_t[1:] / alpha_t[:-1], 0.0001, 0.999)


def linear_beta_schedule(timesteps, beta_min: float = 1e-4, beta_max: float = 0.02):
    return torch.linspace(beta_min, beta_max, timesteps)
