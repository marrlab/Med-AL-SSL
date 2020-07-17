import torch.nn as nn

"""
Autoencoder implementation with Lenet as encoder

Courtesy to: 
https://github.com/afrozalm/AutoEncoder/blob/master/AutoEncoder.lua
"""


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, *self.shape)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class LenetAutoencoder(nn.Module):
    def __init__(self, num_channels, latent_dim=120):
        super(LenetAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(16*7*7, latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(120, 16*7*7),
            nn.Tanh(),
            View((16, 7, 7)),
            nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2, padding=2), # 13
            nn.ReLU(True),
            nn.ConvTranspose2d(6, num_channels, kernel_size=3, stride=2, padding=0, dilation=3, output_padding=1), # 31
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embedding_generator(self, x):
        x = self.encoder(x)
        return x
