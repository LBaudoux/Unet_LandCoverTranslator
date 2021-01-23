import torch
import torch.nn as nn
from graphs.models.custom_layers.double_conv import DoubleConv


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mode='maxpool',factor=2):
        super().__init__()

        if mode=='maxpool':
            self.downsize = nn.Sequential(
                nn.MaxPool2d(factor),
                DoubleConv(in_channels, out_channels)
            )
        elif mode =='avgpool':
            self.downsize = nn.Sequential(
                nn.AvgPool2d(factor),
                DoubleConv(in_channels, out_channels)
            )
        elif mode == 'strideconv':
            self.downsize = nn.Sequential(
                nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,stride=factor),
                DoubleConv(in_channels, out_channels)
            )
        else :
            raise ValueError("Unknown downsizing mode : "+str(mode))

    def forward(self, x):
        return self.downsize(x)