import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels=1, lat_vector=100, feat_map=128):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(lat_vector, feat_map * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.ReLU(True),
            # state size. (feat_map*8) x 4 x 4

            nn.ConvTranspose2d(feat_map * 8, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.ReLU(True),
            # state size. (feat_map*4) x 8 x 8

            nn.ConvTranspose2d( feat_map * 4, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.ReLU(True),
            # state size. (feat_map*2) x 16 x 16

            nn.ConvTranspose2d( feat_map * 2, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.ReLU(True),
            # state size. (feat_map) x 32 x 32

            nn.ConvTranspose2d( feat_map, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
