import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .dataset_utils import WeaklySupervisedDataset
from utils import TransformsSimCLR, TransformFix


class Cifar10Dataset:
    def __init__(self, root, labeled_ratio, add_labeled_ratio, advanced_transforms=True, remove_classes=False,
                 expand_labeled=0, expand_unlabeled=0):
        self.root = root
        self.labeled_ratio = labeled_ratio
        self.cifar_mean = (0.4914, 0.4822, 0.4465)
        self.cifar_std = (0.2023, 0.1994, 0.2010)
        self.expand_labeled = expand_labeled
        self.expand_unlabeled = expand_unlabeled

        self.input_size = 32

        if advanced_transforms:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_mean, self.cifar_std),
            ])

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_mean, self.cifar_std),
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.transform_autoencoder = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.transform_simclr = TransformsSimCLR(size=32)
        self.transform_fixmatch = TransformFix(mean=self.cifar_mean, std=self.cifar_std)
        self.num_classes = 10
        self.add_labeled_ratio = add_labeled_ratio
        self.add_labeled_num = None
        self.remove_classes = remove_classes
        self.classes_to_remove = np.array([0, 1, 2])

    def get_dataset(self):
        base_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True,
                                                    download=True, transform=None)

        self.add_labeled_num = int(len(base_dataset) * self.add_labeled_ratio)

        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(base_dataset)),
            test_size=(1 - self.labeled_ratio),
            shuffle=True,
            stratify=None)

        test_dataset = torchvision.datasets.CIFAR10(root=self.root, train=False,
                                                    download=True, transform=self.transform_test)

        targets = np.array(base_dataset.targets)[labeled_indices]

        if self.remove_classes:
            labeled_indices = labeled_indices[~np.isin(targets, self.remove_classes)]

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=self.transform_train)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_test)

        return base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset

    def get_base_dataset_autoencoder(self):
        base_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True,
                                                    download=True, transform=self.transform_autoencoder)

        return base_dataset

    def get_base_dataset_simclr(self):
        base_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True,
                                                    download=True, transform=self.transform_simclr)

        return base_dataset

    def get_datasets_fixmatch(self, base_dataset, labeled_indices, unlabeled_indices):
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=self.input_size,
                                  padding=int(self.input_size * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar_mean, std=self.cifar_std)
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
