import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut_layer = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + shortcut


class IResNet(nn.Module):
    def __init__(self, block, layers, dropout=0.4, num_features=512):
        super(IResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.body = self._make_layers(block, layers)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, num_features),
            nn.BatchNorm1d(num_features)
        )

    def _make_layers(self, block, layers):
        blocks = []
        in_channels = 64
        channels_cfg = [64, 128, 256, 512]
        for i in range(4):
            stride = 2
            for j in range(layers[i]):
                if j != 0:
                    stride = 1
                blocks.append(block(in_channels, channels_cfg[i], stride))
                in_channels = channels_cfg[i]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


def iresnet100(pretrained=False):
    return IResNet(BasicBlock, [3, 13, 30, 3])
