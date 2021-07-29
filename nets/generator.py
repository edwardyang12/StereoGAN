import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels=1, lat_vector=100, feat_map=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 4, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map) x 32 x 32
            nn.Conv2d(feat_map * 8, feat_map * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

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
