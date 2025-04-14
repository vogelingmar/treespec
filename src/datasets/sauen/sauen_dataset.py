"""Sauen Dataset"""

from typing import Optional
import torch
from torch.utils import data
from torchvision import datasets  # type: ignore
from torchvision.transforms.v2 import Transform  # type: ignore

import pytorch_lightning as L


class SauenDataset(L.LightningDataModule):
    r"""
    Sauen Dataset Class.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
    """

    def __init__(
        self,
        data_dir: str = "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2",
        batch_size: int = 5,
        num_workers: int = 27,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        r"""
        Downloads the dataset to the data_dir if not already present there.
        """

        # add download path for the images of the dataset
        pass

    def setup(self, transform: Optional[Transform] = None):
        r"""
        Creates training (80%), validation (10%) and testing (10%) datasets from the folder structure at data_dir.

        Args:
            transform: Default transformations to be applied to the images.
        """

        # create image folder dataset from images
        self.dataset = datasets.ImageFolder(self.data_dir, transform=transform)

        # calculation of different set sizes (80% traning, 10% validation, 10% test)
        total_size = len(self.dataset)
        val_size = int(0.1 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size

        self.train, self.val, self.test = data.random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self, augmentation: Optional[Transform] = None):
        r"""
        Applies data augmentations to the training dataset and returns a dataloader for the training set.

        Args:
            augmentation: Data augmentations to be applied to the training dataset.
        """

        self.train.dataset.transform = augmentation

        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        r"""
        Returns a dataloader for the validation subset of the dataset.
        """

        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        r"""
        Returns a dataloader for the testing subset of the dataset.
        """

        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def loss_weights(self):
        r"""
        Returns a tensor of weights for the different classes of the dataset to balance training.
        """

        class_counts = torch.bincount(torch.tensor(self.dataset.targets))

        return torch.tensor(1 / class_counts)
