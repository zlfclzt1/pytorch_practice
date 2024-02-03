import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()

    def forward(self, x):
        pass


class Stem(nn.Module):
    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels,
                              kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Body(nn.Module):
    def __init__(self):
        super(Body, self).__init__()

    def forward(self, x):
        pass


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()

    def forwar(self):
        pass


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group_width, b_ratio=1.0, stride=1):
        super(Block, self).__init__()
        b_channels = int(in_channels * b_ratio)
        groups = b_channels // group_width

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, b_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(b_channels)

        self.conv2 = nn.Conv2d(b_channels, b_channels, kernel_size=3,
                               padding=1, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(b_channels)

        self.conv3 = nn.Conv2d(b_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=2)
            self.bn_skip = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

    def forward(self, input):
        # input: N, in_channels, H, W
        residual = input

        # 1 * 1 conv: (in_channels * b_ratio)
        x = self.relu(self.bn1(self.conv1(input)))

        # 3 * 3 conv: (in_channels * b_ratio)
        x = self.relu(self.bn2(self.conv2(x)))

        # 1 * 1 conv: out_channels
        x = self.relu(self.bn3(self.conv3(x)))

        if self.skip is not None:
            skip = self.relu(self.bn_skip(self.skip(residual)))
        else:
            skip = residual

        return x + skip


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.relu(self.conv1(x))
        x = self.softmax(self.conv2(x))

        return input * x


if __name__ == "__main__":
    block1 = Block(in_channels=32, out_channels=32, b_ratio=0.5,
                   group_width=16, stride=1)
    block2 = Block(in_channels=32, out_channels=64, b_ratio=0.5,
                   group_width=16, stride=2)
    se_block = SEBlock(in_channels=64, reduction=2)

    input = torch.randn((4, 32, 16, 16))
    x = block1(input)
    x = block2(x)
    x = se_block(x)
