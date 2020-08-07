import torch.nn as nn

"""
Lenet implementation based on the paper: Gradient Based Learning Applied to Document Recognition
Yann LeCun, Leon Bottou, Yoshua Bengio and Patrick Haffner 
(http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

Courtesy to: https://github.com/aliechoes/pytorch-multichannel-image-classification/blob/master/models/lenet.py
"""


class LeNet(nn.Module):
    def __init__(self, num_channels, num_classes, droprate=0.5, input_size=32):
        super(LeNet, self).__init__()
        self.feat_size = {32: 7, 64: 15}
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*self.feat_size[input_size]*self.feat_size[input_size], 120),
            nn.ReLU(),
            nn.Dropout(p=droprate, inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=droprate, inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        x = self.classifier(feat)
        return x, feat
