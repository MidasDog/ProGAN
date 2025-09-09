import os

import torch
import torch.nn as nn
import torchvision

import config

# implementing WGAN-GP loss
def gradient_penalty(
    critic,
    real,
    fake,
    alpha,
    step,
    lambda_gp = config.LAMBDA_GP,
    device="cpu"
):
    BATCH_SIZE, C, H, W = real.shape
    rand_epsilon = torch.rand(BATCH_SIZE, 1, 1, 1).repeat(1, C, H, W).to(device)
    interpolated_images = rand_epsilon * real + (1 - rand_epsilon) * fake.detach()
    interpolated_images.requires_grad_(True)

    score = critic(interpolated_images, alpha, step)

    gradient = torch.autograd.grad(
        outputs=score,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(score),
        retain_graph=True,
        create_graph=True
    )[0] # gradient of the score, only per batch
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = lambda_gp * torch.mean((gradient_norm - 1)**2)

    return gradient_penalty

def plot_to_tensorboard(
        writer,
        loss_critic,
        loss_gen,
        real,
        fake,
        tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)