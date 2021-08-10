import torch.nn as nn
import torch
"""
class Generator(nn.Module):
    def __init__(self, channels=1, feat_map=16):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(channels, feat_map, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.ReLU(True),

            nn.Conv2d(feat_map, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

"""
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding =1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class ConvTranspose(nn.Module):
        def __init__(self, in_channels, out_channels,stride=2, kernel_size=4):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding =1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.double_conv(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.input = ConvBlock(1,64)
        self.sInput = ConvBlock(64,64, stride=2)
        self.down_1 = ConvBlock(64,128)
        self.sDown_1 = ConvBlock(128,128, stride=2)
        self.down_2 = ConvBlock(128,256)

        self.up_sample_2 = ConvTranspose(256,128)
        self.up_2 = ConvBlock(256,128)

        self.up_sample_3 = ConvTranspose(128,64)
        self.up_3 = ConvBlock(128,64)

        self.temp_4 = nn.Conv2d(64,1, kernel_size=3, padding =1)


    def forward(self,x):
        x1 = self.input(x)
        x2 = self.sInput(x1)
        x3 = self.down_1(x2)
        x4 = self.sDown_1(x3)
        x5 = self.down_2(x4)

        x5 = self.up_sample_2(x5)

        x5 = torch.cat((x3,x5),dim=1)
        x5 = self.up_2(x5)
        x6 = self.up_sample_3(x5)

        x6 = torch.cat((x1,x6),dim=1)
        x6 = self.up_3(x6)
        x7 = nn.Tanh()(self.temp_4(x6))

        return x7
"""


class Generator(nn.Module):
    def __init__(self, channels=1, feat_map=8):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(channels, feat_map, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.ReLU(True),
            
            nn.Conv2d(feat_map, feat_map * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.ReLU(True),
            
            nn.Conv2d(feat_map * 2, feat_map * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.ReLU(True),
            
            nn.Conv2d(feat_map * 4, feat_map * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.ReLU(True),
            
            # state size. (feat_map) x 32 x 32
            nn.Conv2d(feat_map * 8, feat_map * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.ReLU(True),

            nn.Conv2d(feat_map * 4, feat_map * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.ReLU(True),

            nn.Conv2d(feat_map * 2, feat_map, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.ReLU(True),

            nn.Conv2d(feat_map, 1, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

