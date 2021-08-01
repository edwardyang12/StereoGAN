import torch.nn as nn

# going down
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.single_conv(x)

# going up
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.single_conv(x)

class Generator(nn.Module):
    def __init__(self, channels=1, lat_vector=100, feat_map=64):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.down1 = ConvBlock(channels,feat_map)
        self.down2 = ConvBlock(feat_map,feat_map*2)
        self.down3 = ConvBlock(feat_map*2,feat_map*4)
        self.down4 = ConvBlock(feat_map*4,feat_map*8)
        self.down5 = ConvBlock(feat_map*8,feat_map*16)

        self.up1 = ConvTransposeBlock(feat_map*16, feat_map*8)
        self.up2 = ConvTransposeBlock(feat_map*8, feat_map*4)
        self.up3 = ConvTransposeBlock(feat_map*4, feat_map*2)
        self.up4 = ConvTransposeBlock(feat_map*2, feat_map)

        self.up5 = nn.ConvTranspose2d(feat_map, channels, 4, 2, 1, bias=False)
        self.out = nn.Tanh()
        # state size. (channels) x 64 x 64


    def forward(self, input):
        x = self.down1(input)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.out(x)

        return x
