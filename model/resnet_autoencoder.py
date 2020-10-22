import torch
from torch import nn
import torch.nn.functional as F
import math

"""
Resnet and Unet based autoencoder
Resnet18 encoder and Unet decoder

Resnet: https://arxiv.org/abs/1512.03385
Unet: https://arxiv.org/abs/1505.04597
"""


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UnetDec(nn.Module):
    def __init__(self, nc=3, z_dim=128, input_size=32):
        super(UnetDec, self).__init__()

        self.linear = nn.Linear(z_dim, 512)
        self.input_size = input_size
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)
        self.outc = nn.ConvTranspose2d(64, nc, kernel_size=3, stride=int(math.log2(input_size) - 4), padding=1,
                                       bias=False)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        x = F.pad(x, [self.input_size - x.size(2), 0, self.input_size - x.size(3), 0])
        return x


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_blocks=None, z_dim=128, nc=3, input_size=32):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=int(math.log2(input_size) - 4), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, basic_block_enc, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [basic_block_enc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class ResNet18Dec(nn.Module):
    def __init__(self, num_blocks=None, z_dim=128, nc=3, input_size=32):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)
        self.input_size = input_size

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=int(input_size/32))

    def _make_layer(self, basic_block_dec, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [basic_block_dec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = x.view(x.size(0), 3, self.input_size, self.input_size)

        return x


class ResnetAutoencoder(nn.Module):
    def __init__(self, z_dim, drop_rate, num_classes, input_size=32):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim, input_size=input_size)
        self.decoder = ResNet18Dec(z_dim=z_dim, input_size=input_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(z_dim, 84),
            nn.ReLU(),
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def forward_encoder_classifier(self, x):
        z = self.encoder(x)
        x = self.classifier(z)
        return x

    def forward_encoder(self, x):
        z = self.encoder(x)
        return z

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x
