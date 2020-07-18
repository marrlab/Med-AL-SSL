import torch.nn as nn
from utils import Flatten, View

"""
Autoencoder implementation with Lenet as encoder

Courtesy to: 
https://github.com/afrozalm/AutoEncoder/blob/master/AutoEncoder.lua
"""


class LenetAutoencoder(nn.Module):
    def __init__(self, num_channels, num_classes, drop_rate, latent_dim=512):
        super(LenetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*7*7, latent_dim),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16*7*7),
            nn.ReLU(),
            View((16, 7, 7)),
            nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(6, num_channels, kernel_size=3, stride=2, padding=0, dilation=3, output_padding=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(latent_dim, 84),
            nn.ReLU(),
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_classifier(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
