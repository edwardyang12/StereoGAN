import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.input = ConvBlock(1,64)
        self.max_pool_1 = nn.MaxPool2d(2)
        self.down_1 = ConvBlock(64,128)
        self.max_pool_2 = nn.MaxPool2d(2)
        self.down_2 = ConvBlock(128,256)
        self.max_pool_3 = nn.MaxPool2d(2)
        self.bottom = ConvBlock(256,512)
        self.temp_1 = nn.Conv2d(512,256, kernel_size=3, padding=1)
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.up_1 = ConvBlock(512,256)
        self.temp_2 = nn.Conv2d(256,128, kernel_size=3, padding=1)
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.up_2 = ConvBlock(256,128)
        self.temp_3 = nn.Conv2d(128,64, kernel_size=3, padding=1)
        self.up_sample_3 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.up_3 = ConvBlock(128,64)
        self.temp_4 = nn.Conv2d(64,1, kernel_size=3, padding=1 )


    def forward(self,x):
        x1 = self.input(x)
        x2 = self.max_pool_1(x1)
        x2 = self.down_1(x2)
        x3 = self.max_pool_2(x2)
        x3 = self.down_2(x3)
        x3_1 = self.max_pool_3(x3)
        x3_1 = self.bottom(x3_1)
        x4 = nn.ReLU()(self.temp_1(x3_1))
        x4 = self.up_sample_1(x4)
        x4 = torch.cat((x3,x4),dim=1)
        x4 = self.up_1(x4)
        x5 = nn.ReLU()(self.temp_2(x4))
        x5 = self.up_sample_2(x5)
        x5 = torch.cat((x2,x5),dim=1)
        x5 = self.up_2(x5)
        x6 = nn.ReLU()(self.temp_3(x5))
        x6 = self.up_sample_3(x6)
        x6 = torch.cat((x1,x6),dim=1)
        x6 = self.up_3(x6)
        x7 = nn.Tanh()(self.temp_4(x6))

        return x7
