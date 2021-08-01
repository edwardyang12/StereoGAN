import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.single_conv(x)

class Generator(nn.Module):
    def __init__(self, channels=1, lat_vector=100, feat_map=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            ConvBlock(channels,feat_map),
            ConvBlock(feat_map,feat_map*2),
            ConvBlock(feat_map*2,feat_map*4),
            ConvBlock(feat_map*4,feat_map*8),
            ConvBlock(feat_map*8,feat_map*16),

            nn.ConvTranspose2d( feat_map * 16, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.ReLU(True),
            # state size. (feat_map) x 32 x 32

            nn.ConvTranspose2d( feat_map * 8, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( feat_map * 4, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( feat_map * 2, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.ReLU(True),

            nn.ConvTranspose2d( feat_map, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
