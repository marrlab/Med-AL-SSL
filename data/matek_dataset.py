import os
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .dataset_utils import WeaklySupervisedDataset
from utils import TransformsSimCLR


class MatekDataset:
    def __init__(self, root, labeled_ratio, add_labeled_ratio, advanced_transforms=True, remove_classes=False):
        self.root = root
        self.train_path = os.path.join(self.root, "matek", "train")
        self.test_path = os.path.join(self.root, "matek", "test")
        self.labeled_ratio = labeled_ratio
        self.matek_mean = (0.8205, 0.7279, 0.8360)
        self.matek_std = (0.1719, 0.2589, 0.1042)
        self.input_size = 128

        if advanced_transforms:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(self.input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.matek_mean, std=self.matek_std)
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(size=self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.matek_mean, std=self.matek_std)
            ])

        else:
            self.transform_train = transforms.Compose([
                transforms.Resize(size=self.input_size),
                transforms.ToTensor(),
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(size=self.input_size),
                transforms.ToTensor(),
            ])
        self.transform_autoencoder = transforms.Compose([
                transforms.Resize(size=self.input_size),
                transforms.ToTensor(),
            ])
        self.transform_simclr = TransformsSimCLR(size=self.input_size)
        self.num_classes = 15
        self.add_labeled_ratio = add_labeled_ratio
        self.add_labeled_num = None
        self.remove_classes = remove_classes
        self.classes_to_remove = np.array([0, 1, 2, 3, 4, 6, 7, 9, 11, 13, 14])

    def get_dataset(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=None
        )

        self.add_labeled_num = int(len(base_dataset) * self.add_labeled_ratio)

        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(base_dataset)),
            test_size=(1 - self.labeled_ratio),
            shuffle=True,
            stratify=None)

        test_dataset = torchvision.datasets.ImageFolder(
            self.test_path, transform=self.transform_test
        )

        targets = np.array(base_dataset.targets)[labeled_indices]

        if self.remove_classes:
            labeled_indices = labeled_indices[~np.isin(targets, self.remove_classes)]

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=self.transform_train)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_test)

        return labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset

    def get_base_dataset_autoencoder(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=self.transform_autoencoder
        )

        return base_dataset

    def get_base_dataset_simclr(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=self.transform_simclr
        )

        return base_dataset
