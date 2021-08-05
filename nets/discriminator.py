import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=3, feat_map=128):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (channels) x 64 x 64
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map) x 32 x 32
            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map*2) x 16 x 16
            nn.Conv2d(feat_map * 2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map*4) x 8 x 8
            nn.Conv2d(feat_map * 4, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 8, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
