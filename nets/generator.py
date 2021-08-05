import torch.nn as nn
import torch

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

class MiniUnet(nn.Module):
    def __init__(self):
        super(MiniUnet,self).__init__()
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

class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding =1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding =1),
            nn.Tanh()
        )

    def forward(self,x):
        return self.main(x)
