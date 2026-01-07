import torch

from DDPM.diffusion import Diffusion1d
from DDPM.schedulers import CosineScheduler
from DDPM.models import Diffusion1dDefaultModel

DATA_SIZE = 5
NUM_FEATURES = 10

train_data = torch.randn(DATA_SIZE, NUM_FEATURES)

diffusion_scheduler = CosineScheduler()
diffusion_model = Diffusion1d(num_features=NUM_FEATURES, scheduler=diffusion_scheduler)
denoiser = Diffusion1dDefaultModel(num_features=NUM_FEATURES)

diffused, ts, noise = diffusion_model(train_data)

print("-" * 50)
print("Diffused train data:")
print(diffused)

print("-" * 50)
print("Chosen timesteps:")
print(ts)

print("-" * 50)
print("Added noise:")
print(noise)

sample = diffusion_model.sample(denoiser_model=denoiser)
print("-" * 50)
print("Sample")
print(sample)
