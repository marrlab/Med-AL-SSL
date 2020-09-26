import torch.nn as nn
import torchvision
import math


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLRArch(nn.Module):
    def __init__(self, num_channels, num_classes, drop_rate,
                 normalize, latent_dim=64, projection_dim=64,
                 arch='lenet', input_size=32):
        super(SimCLRArch, self).__init__()

        self.normalize = normalize

        if arch == 'lenet':
            self.encoder = nn.Sequential(
                nn.Conv2d(num_channels, 6, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16 * 7 * 7, latent_dim),
                nn.ReLU(),
            )
        else:
            self.encoder = modify_resnet_model(
                torchvision.models.resnet18(), v1=True, input_size=input_size
            )
            latent_dim = self.encoder.fc.in_features
            self.encoder.fc = Identity()

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(latent_dim, 84),
            nn.ReLU(),
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z

    def forward_encoder_classifier(self, x):
        h = self.encoder(x)
        x = self.classifier(h)
        return x

    def forward_encoder(self, x):
        x = self.encoder(x)
        return x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x


import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


def modify_resnet_model(model, *, v1=True, input_size=32):
    """Modifies some layers of a given torchvision resnet model to
    match the one used for the CIFAR-10 experiments in the SimCLR paper.
    Parameters
    ----------
    model : ResNet
        Instance of a torchvision ResNet model.
    cifar_stem : bool
        If True, adapt the network stem to handle the smaller CIFAR images, following
        the SimCLR paper. Specifically, use a smaller 3x3 kernel and 1x1 strides in the
        first convolution and remove the max pooling layer.
    v1 : bool
        If True, modify some convolution layers to follow the resnet specification of the
        original paper (v1). torchvision's resnet is v1.5 so to revert to v1 we switch the
        strides between the first 1x1 and following 3x3 convolution on the first bottleneck
        block of each of the 2nd, 3rd and 4th layers.
    Returns
    -------
    Modified ResNet model.
    :param model:
    :param v1:
    :param input_size:
    """
    assert isinstance(model, ResNet), "model must be a ResNet instance"

    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=int(math.log2(input_size) - 4), padding=1, bias=False)
    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
    model.conv1 = conv1
    model.maxpool = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, "layer{}".format(l))
            block = list(layer.children())[0]
            if isinstance(block, Bottleneck):
                assert block.conv1.kernel_size == (1, 1) and block.conv1.stride == (1, 1)
                assert block.conv2.kernel_size == (3, 3) and block.conv2.stride == (2, 2)
                assert block.conv2.dilation == (1, 1), "Currently, only models with dilation=1 are supported"
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model
