import torch
import torch.nn as nn
import torch.nn.functional as F
from residual_block import ResidualBlock

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        num_input_channels: Number of channel for input images (HST/HSC are 1)
        base_channels: number of channels throughout the generator, a scalar
        n_ps_blocks: number of PixelShuffle blocks, a scalar
        n_res_blocks: number of residual blocks, a scalar
    '''

    def __init__(self,num_input_channels=1, base_channels=64, n_ps_blocks=2, n_res_blocks=16):
        super().__init__()
        # Input layer - take a N channels image and projects it into base channels
        self.in_layer = nn.Sequential(
            nn.Conv2d(num_input_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        # B Residual blocks as shown in the above architecture
        # We defined ResidualBlock Above
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(base_channels)]

        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # PixelShuffle blocks
        ps_blocks = []
        for i in range(n_ps_blocks):
            if i == 0:
                pix_shuffle = 3
                ps_blocks += [
                nn.Conv2d(base_channels, 9 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(pix_shuffle),
                nn.PReLU(),]
            else:
                pix_shuffle = 2
                ps_blocks += [
                nn.Conv2d(base_channels, 4 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(pix_shuffle),
                nn.PReLU(),
            ]
            
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # Output layer
        self.out_layer = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            # nn.Tanh(),
            nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, x):
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.out_layer(x)
        return x