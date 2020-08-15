import os
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .dataset_utils import WeaklySupervisedDataset
from utils import TransformsSimCLR, TransformFix


class MatekDataset:
    def __init__(self, root, labeled_ratio, add_labeled_ratio, advanced_transforms=True, remove_classes=False,
                 expand_labeled=0, expand_unlabeled=0, unlabeled_subset_ratio=1):
        self.root = root
        self.train_path = os.path.join(self.root, "matek", "train")
        self.test_path = os.path.join(self.root, "matek", "test")
        self.labeled_ratio = labeled_ratio
        self.matek_mean = (0.8205, 0.7279, 0.8360)
        self.matek_std = (0.1719, 0.2589, 0.1042)
        self.input_size = 128
        self.expand_labeled = expand_labeled
        self.expand_unlabeled = expand_unlabeled

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
        self.transform_fixmatch = TransformFix(mean=self.matek_mean, std=self.matek_std, input_size=self.input_size)
        self.num_classes = 15
        self.add_labeled_ratio = add_labeled_ratio
        self.unlabeled_subset_ratio = unlabeled_subset_ratio
        self.add_labeled_num = None
        self.unlabeled_subset_num = None
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

        self.unlabeled_subset_num = int(len(unlabeled_indices) * self.unlabeled_subset_ratio)

        test_dataset = torchvision.datasets.ImageFolder(
            self.test_path, transform=self.transform_test
        )

        targets = np.array(base_dataset.targets)[labeled_indices]

        if self.remove_classes:
            labeled_indices = labeled_indices[~np.isin(targets, self.remove_classes)]

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=self.transform_train)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_test)

        return base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset

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

    def get_datasets_fixmatch(self, base_dataset, labeled_indices, unlabeled_indices):
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=self.input_size,
                                  padding=int(self.input_size * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.matek_mean, std=self.matek_std)
        ])

        expand_labeled = self.expand_labeled // len(labeled_indices)
        expand_unlabeled = self.expand_unlabeled // len(unlabeled_indices)
        labeled_indices = np.hstack(
            [labeled_indices for _ in range(expand_labeled)])
        unlabeled_indices = np.hstack(
            [unlabeled_indices for _ in range(expand_unlabeled)])

        if len(labeled_indices) < self.expand_labeled:
            diff = self.expand_labeled - len(labeled_indices)
            labeled_indices = np.hstack(
                (labeled_indices, np.random.choice(labeled_indices, diff)))
        else:
            assert len(labeled_indices) == self.expand_labeled

        if len(unlabeled_indices) < self.expand_unlabeled:
            diff = self.expand_unlabeled - len(unlabeled_indices)
            unlabeled_indices = np.hstack(
                (unlabeled_indices, np.random.choice(unlabeled_indices, diff)))
        else:
            assert len(unlabeled_indices) == self.expand_unlabeled

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=transform_labeled)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_fixmatch)

        return labeled_dataset, unlabeled_dataset
