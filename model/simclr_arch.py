import torch.nn as nn
from utils import Flatten
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLRArch(nn.Module):
    def __init__(self, num_channels, num_classes, drop_rate,
                 normalize, latent_dim=128, projection_dim=128,
                 arch='lenet'):
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
                Flatten(),
                nn.Linear(16 * 7 * 7, latent_dim),
                nn.ReLU(),
            )
        else:
            self.encoder = torchvision.models.resnet18()
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

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x

    def forward_encoder(self, x):
        x = self.encoder(x)
        return x
