import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class ConvTranpose(nn.Module):
        def __init__(self, in_channels, out_channels,stride=2):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.ConvTranpose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.double_conv(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.input = ConvBlock(1,64)
        self.down_1 = ConvBlock(64,128)
        self.down_2 = ConvBlock(128,256)

        self.up_sample_2 = ConvTranpose(256,128)
        self.up_2 = ConvBlock(256,128,stride=1)

        self.up_sample_3 = ConvTranpose(128,64)
        self.up_3 = ConvBlock(128,64,stride=1)

        self.temp_4 = nn.Conv2d(64,1, kernel_size=3, padding=1 )


    def forward(self,x):
        x2 = self.input(x)
        x2 = self.down_1(x2)
        x3 = self.down_2(x2)

        x5 = self.up_sample_2(x3)

        x5 = torch.cat((x2,x5),dim=1)
        x5 = self.up_2(x5)

        x6 = self.up_sample_3(x5)

        x6 = torch.cat((x1,x6),dim=1)
        x6 = self.up_3(x6)
        x7 = nn.Tanh()(self.temp_4(x6))

        return x7
