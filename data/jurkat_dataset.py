import os
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .dataset_utils import WeaklySupervisedDataset
from utils import TransformsSimCLR, TransformFix, oversampling_indices, merge


class JurkatDataset:
    def __init__(self, root, labeled_ratio=1, add_labeled_ratio=0, advanced_transforms=True, remove_classes=False,
                 expand_labeled=0, expand_unlabeled=0, unlabeled_subset_ratio=1, oversampling=True, stratified=False,
                 merged=True, unlabeled_augmentations=False):
        self.root = root
        self.train_path = os.path.join(self.root, "jurkat", "train")
        self.test_path = os.path.join(self.root, "jurkat", "test")
        self.labeled_ratio = labeled_ratio
        self.jurkat_mean = (0, 0, 0)
        self.jurkat_std = (1, 1, 1)
        self.input_size = 32
        self.crop_size = 50
        self.expand_labeled = expand_labeled
        self.expand_unlabeled = expand_unlabeled
        self.oversampling = oversampling
        self.stratified = stratified
        self.merged = merged
        self.merge_classes = [['G1', 'G2', 'S']]

        if advanced_transforms:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(self.crop_size),
                transforms.RandomAffine(degrees=90, translate=(0.2, 0.2)),
                transforms.Resize(size=self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.jurkat_mean, std=self.jurkat_std),
                transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 0.9))
            ])
            self.transform_test = transforms.Compose([
                transforms.Resize(size=self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.jurkat_mean, std=self.jurkat_std)
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
            transforms.RandomCrop(self.crop_size),
            transforms.RandomAffine(degrees=90, translate=(0.2, 0.2)),
            transforms.Resize(size=self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.jurkat_mean, std=self.jurkat_std)
        ])
        self.transform_simclr = TransformsSimCLR(size=self.input_size)
        self.transform_fixmatch = TransformFix(mean=self.jurkat_mean, std=self.jurkat_std, input_size=self.input_size)
        self.merged_classes = 2 if self.merged else 0
        self.num_classes = 7 - self.merged_classes
        self.add_labeled_ratio = add_labeled_ratio
        self.unlabeled_subset_ratio = unlabeled_subset_ratio
        self.add_labeled_num = None
        self.unlabeled_subset_num = None
        self.remove_classes = remove_classes
        self.unlabeled_augmentations = unlabeled_augmentations
        self.labeled_class_samples = None
        self.classes_to_remove = np.array([1, 2, 5])

    @staticmethod
    def check_file_jurkat(path):
        if '_merged' in path:
            return True
        else:
            return False

    def get_dataset(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=None, is_valid_file=self.check_file_jurkat,
        )

        if self.merged:
            base_dataset.targets, base_dataset.classes, base_dataset.class_to_idx = \
                merge(base_dataset, self.merge_classes)

        self.add_labeled_num = int(len(base_dataset) * self.add_labeled_ratio)

        if self.stratified:
            labeled_indices, unlabeled_indices = train_test_split(
                np.arange(len(base_dataset)),
                test_size=(1 - self.labeled_ratio),
                shuffle=True,
                stratify=base_dataset.targets)
        else:
            indices = np.arange(len(base_dataset))
            np.random.shuffle(indices)
            labeled_indices, unlabeled_indices = indices[:self.add_labeled_num], indices[self.add_labeled_num:]

        self.unlabeled_subset_num = int(len(unlabeled_indices) * self.unlabeled_subset_ratio)

        test_dataset = torchvision.datasets.ImageFolder(
            self.test_path, transform=None, is_valid_file=self.check_file_jurkat,
        )

        if self.merged:
            test_dataset.targets, test_dataset.classes, test_dataset.class_to_idx = \
                merge(test_dataset, self.merge_classes)
        test_dataset = WeaklySupervisedDataset(test_dataset, range(len(test_dataset)), transform=self.transform_test)

        if self.remove_classes:
            targets = np.array(base_dataset.targets)[labeled_indices]
            labeled_indices = labeled_indices[~np.isin(targets, self.remove_classes)]

        self.labeled_class_samples = [np.sum(np.array(base_dataset.targets)[labeled_indices] == i)
                                      for i in range(len(base_dataset.classes))]

        if self.oversampling:
            labeled_indices = oversampling_indices(labeled_indices,
                                                   np.array(base_dataset.targets)[labeled_indices])

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=self.transform_train)

        if self.unlabeled_augmentations:
            unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_train)
        else:
            unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_test)

        return base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset

    def get_base_dataset_autoencoder(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=self.transform_autoencoder, is_valid_file=self.check_file_jurkat,
        )

        return base_dataset

    def get_base_dataset_simclr(self):
        base_dataset = torchvision.datasets.ImageFolder(
            self.train_path, transform=self.transform_simclr, is_valid_file=self.check_file_jurkat,
        )

        return base_dataset

    def get_datasets_fixmatch(self, base_dataset, labeled_indices, unlabeled_indices):
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=self.input_size,
                                  padding=int(self.input_size * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.jurkat_mean, std=self.jurkat_std)
        ])

        expand_labeled = self.expand_labeled // len(labeled_indices)
        expand_unlabeled = self.expand_unlabeled // len(unlabeled_indices)
        labeled_indices = \
            np.hstack([labeled_indices for _ in range(expand_labeled)]) \
            if len(labeled_indices) < self.expand_labeled else labeled_indices
        unlabeled_indices = \
            np.hstack([unlabeled_indices for _ in range(expand_unlabeled)]) \
            if len(unlabeled_indices) < self.expand_unlabeled else unlabeled_indices

        if len(labeled_indices) < self.expand_labeled:
            diff = self.expand_labeled - len(labeled_indices)
            labeled_indices = np.hstack(
                (labeled_indices, np.random.choice(labeled_indices, diff)))

        if len(unlabeled_indices) < self.expand_unlabeled:
            diff = self.expand_unlabeled - len(unlabeled_indices)
            unlabeled_indices = np.hstack(
                (unlabeled_indices, np.random.choice(unlabeled_indices, diff)))

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=transform_labeled)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_fixmatch)

        return labeled_dataset, unlabeled_dataset
