import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        num_input_channels: Number of channel for input images (HST/HSC are 1)

        base_channels: number of channels in first convolutional layer, a scalar
        n_blocks: number of convolutional blocks, a scalar
    '''

    def __init__(self,num_input_channels=1, base_channels=64, n_blocks=3):
        super().__init__()
        self.blocks = [
            nn.Conv2d(num_input_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        cur_channels = base_channels
        for i in range(n_blocks):
            self.blocks += [
                nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(2 * cur_channels, 2 * cur_channels, kernel_size=3, padding=1, stride=2),
                # nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            cur_channels *= 2

        self.blocks += [
            # You can replicate nn.Linear with pointwise nn.Conv2d i.e. kernel size=1
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * cur_channels, 1, kernel_size=1, padding=0),

            # Apply sigmoid if necessary in loss function for stability
            nn.Flatten(),
        ]

        self.layers = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.layers(x)