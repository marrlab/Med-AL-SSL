from torch.utils.data import Dataset
import numpy as np


class WeaklySupervisedDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.transform = transform
        self.indices = indices
        self.dataset = dataset
        self.targets = np.array(dataset.targets)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, target = self.dataset[self.indices[index]]
        img = self.transform(img) if self.transform is not None else img

        return img, target
