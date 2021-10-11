"""
Author: Isabella Liu 10/3/21
Feature: Simple MLP transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .psmnet_submodule import *


class Transformer(nn.Module):
    def __init__(self, inplanes=3):
        super(Transformer, self).__init__()

        self.conv = nn.Sequential(
            # convbn(inplanes, 3, 3, 1, 1, 1),
            # nn.ReLU(inplace=True),
            # convbn(32, 128, 3, 1, 1, 1),
            # nn.ReLU(inplace=True),
            # convbn(3, 3, 3, 1, 1, 1),
            # nn.ReLU(inplace=True),
            # convbn(3, 3, 3, 1, 1, 1),
            # nn.ReLU(inplace=True),
            # convbn(3, 3, 3, 1, 1, 1),
            # nn.ReLU(inplace=True),
            # convbn(3, 3, 3, 1, 1, 1),
            # nn.ReLU(inplace=True),

            convbn(inplanes, 3, 3, 1, 1, 1),
            nn.Sigmoid(),
            convbn(3, 3, 3, 1, 1, 1),
            nn.Sigmoid(),
            convbn(3, 3, 3, 1, 1, 1),
            nn.Sigmoid(),
            convbn(3, 3, 3, 1, 1, 1),
            nn.Sigmoid(),
            convbn(3, 3, 3, 1, 1, 1),
            nn.Sigmoid(),
            convbn(3, 3, 3, 1, 1, 1),
            nn.Tanh(),
        )

    def forward(self, img_L, img_R, img_real_L, img_real_R):
        img_L_transformed = self.conv(img_L)
        img_R_transformed = self.conv(img_R)
        img_real_L_transformed = self.conv(img_real_L)
        img_real_R_transformed = self.conv(img_real_R)
        return img_L_transformed, img_R_transformed, img_real_L_transformed, img_real_R_transformed


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.conv = nn.Sequential(
            convbn(6, 3, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature, image):
        inupt = torch.cat((
            feature,
            image
        ), 1)
        output = self.conv(inupt)
        return output



if __name__ == '__main__':
    # model = Transformer().cuda()
    # imgL = torch.rand(1, 1, 256, 512).cuda()
    # mid_feaL = model(imgL)
    # loss = F.smooth_l1_loss(imgL, mid_feaL)
    # print(mid_feaL.shape)
    # print(loss)
    # loss.backward()

    model = Fusion().cuda()
    img = torch.rand(1, 3, 256, 512).cuda()
    fea = torch.rand(1, 3, 256, 512).cuda()
    output = model(fea, img)
    print(output.shape)
