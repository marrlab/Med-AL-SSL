import torch.nn as nn

"""
Shallow Autoencoder implementation.

Courtesy to: 
https://github.com/aliechoes/pytorch-multichannel-image-feature-extraction/blob/master/machine_learning/models.py
"""


class ShallowAutoencoder(nn.Module):
    def __init__(self, num_channels):
        super(ShallowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, num_channels, 2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embedding_generator(self, x):
        x = self.encoder(x)
        x = x.view(-1,8*2*2)
        return x
