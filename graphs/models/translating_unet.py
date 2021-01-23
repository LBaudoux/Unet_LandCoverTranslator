import torch.nn as nn
import torch
from graphs.models.custom_layers.down_block import Down
from graphs.models.custom_layers.up_block import Up
from graphs.models.custom_layers.double_conv import DoubleConv

class TranslatingUnet(nn.Module):
    def __init__(self, config):
        self.config=config
        n_channels = self.config.input_channels
        n_classes = self.config.num_classes
        number_of_feature_map = self.config.number_of_feature_map
        down_mode = self.config.down_mode
        up_mode=self.config.up_mode
        self.max_feature_map=8 * number_of_feature_map
        self.min_size=self.config.image_size//2//5//2//2
        self.number_of_feature_map=number_of_feature_map
        super().__init__()

        self.inc = DoubleConv(n_channels, number_of_feature_map)
        self.down1 = Down(number_of_feature_map, 2 * number_of_feature_map,mode=down_mode)
        self.down2 = Down(2 * number_of_feature_map, 4 * number_of_feature_map, factor=5,mode=down_mode)
        self.down3 = Down(4 * number_of_feature_map, self.max_feature_map,mode=down_mode)
        self.down4 = Down( self.max_feature_map,self.max_feature_map,mode=down_mode)
        self.up1 = Up(16 * number_of_feature_map, 4 * number_of_feature_map, up_mode)
        self.up2 = Up(self.max_feature_map, 2 * number_of_feature_map, up_mode)
        self.outc = nn.Conv2d(2*number_of_feature_map, n_classes, kernel_size=1)
        self.sea_conv=DoubleConv(1, n_classes)


        self.fc = nn.Linear(128, self.max_feature_map)


    def forward(self, x,sea,coord):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        coord = nn.ReLU(inplace=True)(self.fc(coord))
        batch_size=coord.shape[0]

        x5= x5 + (coord.permute(1,0) * torch.ones((self.min_size,self.min_size,self.max_feature_map,batch_size),device=coord.device)).permute(3,2,1,0)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        sea=self.sea_conv(sea)
        logits = self.outc(x)+sea
        return logits
