from torch.utils.data import Dataset
import numpy as np


class WeaklySupervisedDataset(Dataset):
    def __init__(self, dataset, indices, transform=None, poisson=False):
        self.transform = transform
        self.indices = indices
        self.dataset = dataset
        self.targets = np.array(dataset.targets)
        self.poisson = poisson

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, _ = self.dataset[self.indices[index]]
        target = self.targets[self.indices[index]]

        img = self.transform(img) if self.transform is not None else img
        poisson_noise = np.random.poisson(size=img.shape) if type(img) is not tuple else \
            np.random.poisson(size=img[0].shape)
        img = img + poisson_noise if self.poisson else img

        return img, target
