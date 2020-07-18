import torch.nn as nn
from utils import Flatten, View

"""
A simple Autoencoder architecture.

Courtesy to: https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
"""


class SimpleAutoencoder(nn.Module):
    def __init__(self, num_channels, num_classes, drop_rate, latent_dim=512):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(48 * 4 * 4, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48 * 4 * 4),
            nn.ReLU(True),
            View((48, 4, 4)),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, num_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(latent_dim, 84),
            nn.ReLU(),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def forward_classifier(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
