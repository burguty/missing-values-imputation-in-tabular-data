import torch

from DDPM.diffusion import Diffusion1d
from DDPM.schedulers import CosineScheduler
from DDPM.denoisers import SimpleMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

DATA_SIZE = 5
NUM_FEATURES = 10

train_data = torch.randn(DATA_SIZE, NUM_FEATURES).to(device)

denoiser = SimpleMLP(NUM_FEATURES)
diffusion = Diffusion1d(NUM_FEATURES, timesteps=1000, denoiser=denoiser)
diff_scheduler = CosineScheduler(diffusion)

diffusion.train()

diffused, ts, noise = diffusion(train_data)

print("-" * 50)
print("Diffused train data:")
print(diffused)

print("-" * 50)
print("Chosen timesteps:")
print(ts)

print("-" * 50)
print("Added noise:")
print(noise)

diffusion.eval()

sample = diffusion.sample()
print("-" * 50)
print("Sample")
print(sample)
