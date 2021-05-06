import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values
        channels: the number of channels throughout the residual block, a scalar
        padding_mode: Type of padding to apply to convolution
    '''

    def __init__(self, channels,padding_mode="zeros"):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,padding_mode=padding_mode),
            nn.BatchNorm2d(channels),
            nn.PReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1,padding_mode=padding_mode),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.layers(x)