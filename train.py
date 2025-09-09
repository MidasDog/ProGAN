import os

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Critic
import config
import utils

from math import log2
from tqdm import tqdm

torch.backends.cudnn.benchmarks = True

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)]
            )
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True # since we're working with NVIDIA GPU
    )

    return loader, dataset

def train_fn(
    gen,                    # Generator
    critic,                 # Critic
    loader,                 # Dataloader
    dataset,                # Dataset image folder
    step,                   # Image size step (progressive)
    alpha,                  # Fade_in constant
    opt_critic,             # Critic optimizer
    opt_gen,                # Generator optimizer
    tensorboard_step,       # for tensorboard
    writer,                 # for tensorboard
    gradscaler_gen,
    gradscaler_critic
):
    loop = tqdm(loader)
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: -> min -(E[critic(real)] - E[critic(fake)])
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        with torch.amp.autocast("cuda"):
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = utils.gradient_penalty(critic, real, fake, alpha, step, lambda_gp=config.LAMBDA_GP, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + gp # WGAN-GP loss
                + (0.001 * torch.mean(critic_real**2)) # additional term to keep critic output from drifting too far away from zero (from paper)
            )
        
        opt_critic.zero_grad()
        gradscaler_critic.scale(loss_critic).backward()
        gradscaler_critic.step(opt_critic)
        gradscaler_critic.update()

        # Train Generator -> min -(E[critic(gen_fake)])
        with torch.amp.autocast("cuda"):
            gen_fake_score = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake_score)

        opt_gen.zero_grad()
        gradscaler_gen.scale(loss_gen).backward()
        gradscaler_gen.step(opt_gen)
        gradscaler_gen.update()

        # update alpha and ensure it's always <= 1
        alpha += cur_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            utils.plot_to_tensorboard(writer=writer,
                                      loss_critic=loss_critic.item(),
                                      loss_gen=loss_gen.item(),
                                      real=real.detach(),
                                      fake=fixed_fakes.detach(),
                                      tensorboard_step=tensorboard_step)
            tensorboard_step += 1
            
    return tensorboard_step, alpha
    

def main():
    gen = Generator(z_dim=config.Z_DIM, in_channels=config.IN_CHANNELS, img_channels=3).to(config.DEVICE)
    critic = Critic(in_channels=config.IN_CHANNELS, img_channels=3).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LR, betas=(0.0, 0.99), eps=1e-8)
    opt_critic = optim.Adam(critic.parameters(), lr=config.LR, betas=(0.0, 0.99), eps=1e-8)

    gradscaler_gen = torch.amp.GradScaler()
    gradscaler_critic = torch.amp.GradScaler()
    
    writer = SummaryWriter(f"logs/progan1")

    gen.train()
    critic.train()

    tensorboard_step = 0

    # start at image START_TRAIN_AT_IMG_SIZE
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4)) # convert img_size into step numbers
    for prog_epoch in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4 * 2**step) # img_size at train, inverse of step equation
        print(f"Image size: {4 * 2**step}")

        for epoch in range(prog_epoch):
            print(f"Epoch [{epoch+1}/{prog_epoch}]")
            tensorboard_step, alpha = train_fn(
                gen,
                critic,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                gradscaler_gen,
                gradscaler_critic
            )
        
        step += 1

if __name__ == "__main__":
    main()