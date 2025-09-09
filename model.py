import os

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms, datasets

from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

# Equalized LR
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.register_buffer('scale', torch.tensor((gain / (in_channels * kernel_size**2))**0.5))
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale.to(x.dtype)) + self.bias.view(1, self.bias.shape[0], 1, 1).to(x.dtype)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        eps = torch.tensor(self.epsilon, dtype=x.dtype, device=x.device) # AMP-safe epsilon
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.use_pn = use_pixelnorm
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.pn(x) if self.use_pn else x
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.pn(x) if self.use_pn else x
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(in_channels=z_dim,
                               out_channels=in_channels,
                               kernel_size=4,
                               stride=1,
                               padding=0), # 1x1 -> 4x4 (actually should've used equalized lr for transpose as well)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels=in_channels,
                     out_channels=in_channels,
                     kernel_size=3,
                     stride=1,
                     padding=1), # 4x4 -> 4x4
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        # also known as toRGB in the paper
        self.initial_rgb = WSConv2d(in_channels=in_channels,
                                    out_channels=img_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb])
        )

        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            
            self.prog_blocks.append(
                ConvBlock(conv_in_channels, conv_out_channels)
            )
            # also known as toRGB in the paper
            self.rgb_layers.append(
                WSConv2d(
                    in_channels=conv_out_channels,
                    out_channels=img_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

    def fade_in(self, alpha, upscaled, generated):
        # alpha scalar [0, 1], upscaled.shape == generated.shape
        alpha = torch.tensor(alpha, dtype=generated.dtype, device=generated.device)
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)
    
    def forward(self, x, alpha, steps):
        out = self.initial(x) # 4x4

        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            upscaled_prog = interpolate(out, scale_factor=2, mode="nearest") # Upsample result from init -> upscaled
            out = self.prog_blocks[step](upscaled_prog) # Run through conv block (with proGAN architecture)
            
        final_upscaled = self.rgb_layers[steps - 1](upscaled_prog)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)

class Critic(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Reverse order from Generator architecture
        # Make sure that the shapes are correctly mirrored
        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i-1])
            self.prog_blocks.append(
                ConvBlock(conv_in_channels, conv_out_channels)
            )
            # also known as fromRGB in the paper
            self.rgb_layers.append(
                WSConv2d(
                    in_channels=img_channels,
                    out_channels=conv_in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        # also known as fromRGB in the paper
        self.initial_rgb = WSConv2d(
            in_channels=img_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        ) # self.initial_rgb is used at the end for 4x4

        self.rgb_layers.append(self.initial_rgb) # ModuleList is read from behind (starting from -1)

        self.avg_pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        self.final_block = nn.Sequential(
            WSConv2d(
                in_channels=in_channels+1,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=4,
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0
            ) # Paper uses linear layer instead of equalized lr
        )
    
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(
            x.shape[0], 1, x.shape[2], x.shape[3]
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps): # steps=0 (4x4), steps=1 (8x8), etc
        cur_steps = len(self.prog_blocks) - steps # so we can start from the end of the ModuleList

        out = self.rgb_layers[cur_steps](x) # fromRGB of Output
        out = self.leaky(out) # LeakyRelu layer

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.avg_pool(x) # Downsampling
        downscaled = self.rgb_layers[cur_steps + 1](downscaled) # fromRGB
        downscaled = self.leaky(downscaled) # LeakyRelu layer

        out = self.prog_blocks[cur_steps](out)
        out = self.avg_pool(out)
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_steps + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)

        return self.final_block(out).view(out.shape[0], -1)

# Testing block
# if __name__ == "__main__":
#     Z_DIM = 50
#     IN_CHANNELS = 256
#     gen = Generator(
#         z_dim=Z_DIM,
#         in_channels=IN_CHANNELS,
#         img_channels=3
#     )
#     critic = Critic(
#         in_channels=IN_CHANNELS,
#         img_channels=3
#     )

#     for img_size in [4, 8, 16, 32, 128, 256, 512, 1024]:
#         num_steps = int(log2(img_size / 4))
#         x = torch.randn(1, Z_DIM, 1, 1)

#         z = gen(x, 0.5, num_steps)
#         assert z.shape == (1, 3, img_size, img_size)

#         out = critic(z, 0.5, num_steps)
#         assert out.shape == (1, 1)


#         print(f"Success at img_size: {img_size}")
