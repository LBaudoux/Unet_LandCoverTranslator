import torch
import torch.nn as nn
from graphs.models.custom_layers.double_conv import DoubleConv
from torch.nn.functional import pad
from torch import tensor,cat


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mode='transposeconv'):
        super().__init__()

        if mode=='bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode=='transposeconv':
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        else:
            raise ValueError("Unknown upsizing mode : " + str(mode))

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = tensor([x2.size()[2] - x1.size()[2]])
        diffX = tensor([x2.size()[3] - x1.size()[3]])

        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = cat([x2, x1], dim=1)
        return self.conv(x)