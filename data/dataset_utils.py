from torch.utils.data import Dataset
from torchvision import datasets


class ActiveDataset(Dataset):
    def __init__(self, root, indexes, transform=None):
        self.transform = transform
        self.indexes = indexes
        self.dataset = datasets.ImageFolder(root, transform=self.transform)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img, target = self.dataset[self.indexes[index]]

        return img, target
