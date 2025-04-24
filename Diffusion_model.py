"""
Diffusion model trained to reproduce the hand written digits contained in thh MNIST dataset

    - Trained with score matching technique --> time step embedding 
    - Conditional sampling: I want to determine the number of the geenerated digit --> condition embedding
    
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *

## Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed
seed = 200
torch.manual_seed(seed)
diffusion = Diffusion(T=100, beta_min=1e-4, beta_max = 0.02, device = DEVICE)

new_model = False; training = False # Change to True if you want train a new model

if not new_model:
    # Loading the trained model
    diffusion.model = torch.load('Trained_model.pth',weights_only=False).to(DEVICE)

if training:
    ## Uploading images
    # Set the number of images to load 
    num_imgs = 200 # You can decrease to speed up training 
    digits, labels = Load_images(N = num_imgs)

    ## Splitting in batches
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.tensor(digits).to(DEVICE),torch.tensor(labels).to(DEVICE))
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    num_epochs = 10  # Increase for better results
    lr = 1e-4
    diffusion.training(num_epochs, dataloader, lr)


############################## SAMPLING ##############################
batch = 15
generated_image, label, frames, frames_t = diffusion.sample(batch, 28)

fig, axes = plt.subplots(batch // 3, 3, figsize=(10, 10))
axes = axes.flatten()

for j, ax in enumerate(axes):
    ax.imshow(generated_image[j, :, :])  # Adding cmap='gray' if images are grayscale
    ax.axis("off")
    ax.set_title(f"Label: {int(label[j, 0])}", fontsize=12, fontweight='bold')

fig.suptitle("Generated Images", fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Second plot (frames with corresponding time steps)
fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
axes = axes.flatten()

for j, ax in enumerate(axes):
    ax.imshow(frames[j])  # Adding cmap='gray' if frames are grayscale
    ax.axis("off")
    ax.set_title(f"time step = {frames_t[j]}", fontsize=12, fontweight='bold')

fig.suptitle("Diffusion Process - Denoising", fontsize=16, fontweight='bold', y=0.8)
plt.tight_layout(rect=[0, 0, 1, 0.75])
plt.show()