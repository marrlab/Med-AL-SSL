import os
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .dataset_utils import WeaklySupervisedDataset


class MatekDataset:
    def __init__(self, root, labeled_ratio, add_labeled_ratio):
        self.root = root
        self.train_path = os.path.join(self.root, "matek", "train")
        self.test_path = os.path.join(self.root, "matek", "test")
        self.labeled_ratio = labeled_ratio
        self.matek_mean = (0.8205, 0.7279, 0.8360)
        self.matek_std = (0.1719, 0.2589, 0.1042)
        self.input_size = 32
        self.transform_train = transforms.Compose([
            transforms.Resize(size=self.input_size),
            transforms.RandomCrop(self.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomGrayscale(),
            # transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.matek_mean, std=self.matek_std)
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(size=self.input_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.matek_mean, std=self.matek_std)
        ])
        self.num_classes = 15
        self.add_labeled_ratio = add_labeled_ratio
        self.add_labeled_num = None

    def get_dataset(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=None
        )

        self.add_labeled_num = int(len(base_dataset) * self.add_labeled_ratio)

        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(base_dataset)),
            test_size=(1 - self.labeled_ratio),
            shuffle=True,
            stratify=base_dataset.targets)

        test_dataset = torchvision.datasets.ImageFolder(
            self.test_path, transform=self.transform_test
        )

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=self.transform_train)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_test)

        return labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset
