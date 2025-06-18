"""Sauen Dataset"""

import os

from typing import Optional
import torch
from torch.utils import data
from torchvision import datasets  # type: ignore
from torchvision.transforms.v2 import Transform  # type: ignore

import pytorch_lightning as L


class ImageDataset(L.LightningDataModule):
    r"""
    Image Dataset Class.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.classes = sorted(folder.name for folder in os.scandir(data_dir) if folder.is_dir())

    def prepare_data(self):
        r"""
        Downloads the dataset to the data_dir if not already present there.
        """

        pass  # pylint: disable=unnecessary-pass

    def setup(self, transform: Optional[Transform] = None):  # pylint: disable=arguments-renamed
        r"""
        Creates training (80%), validation (10%) and testing (10%) datasets from the folder structure at data_dir,
        ensuring that all images of the same tree (by tree ID prefix) are in the same split.

        Args:
            transform: Default transformations to be applied to the images.
        """

        # Create the dataset without splitting yet
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)

        # Group image indices by tree ID (assumes filename starts with tree ID, e.g., '1234_img1.jpg')
        from collections import defaultdict
        import os

        tree_to_indices = defaultdict(list)
        for idx, (path, _) in enumerate(full_dataset.samples):
            filename = os.path.basename(path)
            tree_id = filename.split('_')[0]  # Adjust if your separator is different
            tree_to_indices[tree_id].append(idx)

        # Shuffle tree IDs and split into train/val/test
        import random
        tree_ids = list(tree_to_indices.keys())
        random.shuffle(tree_ids)

        total_trees = len(tree_ids)
        val_trees = int(0.1 * total_trees)
        test_trees = int(0.1 * total_trees)
        train_trees = total_trees - val_trees - test_trees

        train_ids = tree_ids[:train_trees]
        val_ids = tree_ids[train_trees:train_trees + val_trees]
        test_ids = tree_ids[train_trees + val_trees:]

        # Collect indices for each split
        train_indices = [idx for tid in train_ids for idx in tree_to_indices[tid]]
        val_indices = [idx for tid in val_ids for idx in tree_to_indices[tid]]
        test_indices = [idx for tid in test_ids for idx in tree_to_indices[tid]]

        # Create Subsets
        self.dataset = full_dataset  # Save for loss_weights
        self.train = data.Subset(full_dataset, train_indices)
        self.val = data.Subset(full_dataset, val_indices)
        self.test = data.Subset(full_dataset, test_indices)

    def train_dataloader(self, augmentation: Optional[Transform] = None):
        r"""
        Applies data augmentations to the training dataset and returns a dataloader for the training set.

        Args:
            augmentation: Data augmentations to be applied to the training dataset.
        """

        self.train.dataset.transform = augmentation  # type: ignore

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
