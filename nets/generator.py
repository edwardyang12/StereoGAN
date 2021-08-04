import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels=3, feat_map=128):
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
