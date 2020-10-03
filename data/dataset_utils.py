from torch.utils.data import Dataset
import numpy as np
from skimage.util import random_noise
import torch


class WeaklySupervisedDataset(Dataset):
    def __init__(self, dataset, indices, transform=None, poisson=False, seed=9999):
        self.transform = transform
        self.indices = indices
        self.dataset = dataset
        self.targets = np.array(dataset.targets)
        self.poisson = poisson
        self.poisson_coefficient = 0.5
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_raw, _ = self.dataset[self.indices[index]]
        target = self.targets[self.indices[index]]

        img_transformed = self.transform(img_raw) if self.transform is not None else img_raw

        if type(img_transformed) is tuple:
            img_noisy_1 = random_noise(img_transformed[0], mode='poisson', seed=self.seed)
            img_noisy_1 = torch.from_numpy(img_noisy_1) if self.poisson else img_transformed[0]

            img_noisy_2 = random_noise(img_transformed[1], mode='poisson', seed=self.seed)
            img_noisy_2 = torch.from_numpy(img_noisy_2) if self.poisson else img_transformed[1]

            return (img_noisy_1, img_noisy_2), target
        else:
            img_noisy = random_noise(img_transformed, mode='poisson', seed=self.seed)
            img_noisy = torch.from_numpy(img_noisy) if self.poisson else img_transformed

            return img_noisy, target
