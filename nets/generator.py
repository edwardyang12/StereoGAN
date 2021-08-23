"""
Author: Isabella Liu 8/8/21
Feature: A simple, shallow generator network
"""

import math
import torch
import torch.nn as nn


class SimpleG(nn.Module):
    def __init__(self):
        super(SimpleG, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, dilation=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Tanh()
        )

    def forward(self, input):
        """
        :param input: [bs, 1, H, W] TODO: enable optional input channel
        :return: [bs, 1, H, W] output of generator
        """
        output = self.net1(input)
        output = self.net2(output)
        output = self.net3(output)
        return output


class DownUpG(nn.Module):
    def __init__(self):
        super(DownUpG, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, dilation=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, dilation=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        :param input: [bs, 1, H, W] TODO: enable optional input channel
        :return: [bs, 1, H, W] output of generator
        """
        output = self.down1(input)
        output = self.down2(output)
        # output = self.up2(output)
        # output = self.up1(output)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1, output_padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUnetG(nn.Module):
    def __init__(self):
        super(SimpleUnetG, self).__init__()
        self.flat1_1 = ConvBlock(1, 16)
        self.down1 = ConvBlock(16, 32, stride=2)
        self.down2 = ConvBlock(32, 64, stride=2)
        self.up2 = ConvTransposeBlock(64, 32, stride=2)
        self.flat2 = ConvBlock(64, 32)
        self.up1 = ConvTransposeBlock(32, 16, stride=2)
        self.flat1_2 = ConvBlock(32, 16)
        self.flat1_3 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        conv1_1 = self.flat1_1(input)
        conv2_1 = self.down1(conv1_1)
        conv3 = self.down2(conv2_1)
        conv2_2 = self.up2(conv3)
        conv2_3 = torch.cat((conv2_1, conv2_2), dim=1)
        conv2_3 = self.flat2(conv2_3)
        conv1_2 = self.up1(conv2_3)
        conv1_3 = torch.cat((conv1_1, conv1_2), dim=1)
        conv1_3 = self.flat1_2(conv1_3)
        output = self.flat1_3(conv1_3)
        return output


if __name__ == '__main__':
    simpleG = SimpleUnetG().cuda()
    input = torch.rand(1, 1, 256, 512).cuda()
    output = simpleG(input)
    print(output.shape)