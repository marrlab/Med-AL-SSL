import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from .dataset_utils import WeaklySupervisedDataset


class Cifar10Dataset:
    def __init__(self, root, labeled_ratio, add_labeled_ratio):
        self.root = root
        self.labeled_ratio = labeled_ratio
        self.cifar_mean = (0.4914, 0.4822, 0.4465)
        self.cifar_std = (0.2023, 0.1994, 0.2010)
        self.input_size = 32
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomGrayscale(),
            # transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar_mean, std=self.cifar_std)
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar_mean, std=self.cifar_std)
        ])
        self.num_classes = 10
        self.add_labeled_ratio = add_labeled_ratio
        self.add_labeled_num = None

    def get_dataset(self):
        base_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True,
                                                    download=True, transform=None)

        self.add_labeled_num = int(len(base_dataset) * self.add_labeled_ratio)

        # TODO: Do unstratified sampling
        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(base_dataset)),
            test_size=(1 - self.labeled_ratio),
            shuffle=True,
            stratify=None)

        test_dataset = torchvision.datasets.CIFAR10(root=self.root, train=False,
                                                    download=True, transform=self.transform_test)

        labeled_dataset = WeaklySupervisedDataset(base_dataset, labeled_indices, transform=self.transform_train)
        unlabeled_dataset = WeaklySupervisedDataset(base_dataset, unlabeled_indices, transform=self.transform_test)

        return labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset
