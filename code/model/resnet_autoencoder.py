import torch
from torch import nn
import torch.nn.functional as F
from model.resnet import resnet18

"""
Resnet and Unet based autoencoder
Resnet18 encoder and Unet decoder

Resnet: https://arxiv.org/abs/1512.03385
Unet: https://arxiv.org/abs/1505.04597
"""


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
        self.encoder = resnet18(num_classes=num_classes, input_size=input_size, drop_rate=drop_rate)
        self.encoder.linear = nn.Linear(512, z_dim)

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

    def forward_features(self, x):
        out, feat_list = self.encoder.forward_features(x)
        out = self.classifier(out)
        return out, feat_list

    def forward_embeddings(self, x):
        out, feat = self.encoder.forward_embeddings(x)
        return out, feat

    def get_embedding_dim(self):
        return self.encoder.embedding_dim
