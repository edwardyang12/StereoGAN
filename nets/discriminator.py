import torch.nn as nn

class OrigDiscriminator(nn.Module):
    def __init__(self, channels=1, feat_map=64):
        super(OrigDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (channels) x 512 x 512
            # input is actually (chennels) x 64 x 64 patches
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

            nn.Conv2d(feat_map * 8, feat_map * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
