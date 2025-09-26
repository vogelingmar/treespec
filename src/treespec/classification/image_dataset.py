"""Image Dataset"""

import os

from collections import defaultdict
from typing import Optional
from pathlib import Path
import random
import torch
from torch.utils import data
from torchvision import datasets  # type: ignore
import torchvision.transforms.v2 as transforms  # type: ignore
from torchvision.transforms.v2 import Transform

import pytorch_lightning as L


class ImageDataset(L.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    r"""
    Image Dataset Class. That creates a classification dataset with training,
    validation and test splits from a folder structure.

    Args:
        data_dir: Path to the dataset directory.
        batch_size: Batch size for data loaders.
        num_workers: Number of workers for data loaders.
        use_ids: If True, uses tree IDs from the beginning of filenames
            to ensure that there is no data leakage between splits.
    """

    def __init__(self, dataset_dir_path: Path, batch_size: int, num_workers: int, use_ids: bool) -> None:
        super().__init__()
        self.data_dir = dataset_dir_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_ids = use_ids

        self.dataset = None
        self.train = None
        self.val = None
        self.test = None

        self.classes = sorted(folder.name for folder in os.scandir(dataset_dir_path) if folder.is_dir())

    def setup(
        self, transform: Optional[Transform] = None
    ) -> None:  # pylint: disable=arguments-renamed, disable=too-many-locals
        r"""
        Creates training (80%), validation (10%), and testing (10%) datasets from the folder structure at data_dir,
        ensuring that all images of the same tree (by tree ID prefix) are in the same split.

        Args:
            transform: Default transformations to be applied to the images.

        Raises:
            ValueError: If the dataset does not contain at least 10 images or 10 unique tree IDs.
        """
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)

        if len(full_dataset.samples) < 10:
            raise ValueError("Dataset must contain at least 10 images.")

        if self.use_ids:
            tree_to_indices = defaultdict(list)
            for idx, (path, _) in enumerate(full_dataset.samples):
                filename = os.path.basename(path)
                tree_id = filename.split("_")[0]
                tree_to_indices[tree_id].append(idx)

            if len(tree_to_indices) < 10:
                raise ValueError("Dataset must contain at least 10 unique tree IDs.")

            tree_ids = list(tree_to_indices.keys())
            random.shuffle(tree_ids)

            total_trees = len(tree_ids)
            val_trees = int(0.1 * total_trees)
            test_trees = int(0.1 * total_trees)
            train_trees = total_trees - val_trees - test_trees

            train_ids = tree_ids[:train_trees]
            val_ids = tree_ids[train_trees : train_trees + val_trees]
            test_ids = tree_ids[train_trees + val_trees :]

            train_indices = [idx for tid in train_ids for idx in tree_to_indices[tid]]
            val_indices = [idx for tid in val_ids for idx in tree_to_indices[tid]]
            test_indices = [idx for tid in test_ids for idx in tree_to_indices[tid]]

            self.dataset = full_dataset
            self.train = data.Subset(full_dataset, train_indices)
            self.val = data.Subset(full_dataset, val_indices)
            self.test = data.Subset(full_dataset, test_indices)

        else:
            self.dataset = datasets.ImageFolder(  # pylint: disable=attribute-defined-outside-init
                root=self.data_dir, transform=transform
            )

            # (80% training, 10% validation, 10% test)
            total_size = len(self.dataset)
            val_size = int(0.1 * total_size)
            test_size = int(0.1 * total_size)
            train_size = total_size - val_size - test_size

            self.train, self.val, self.test = data.random_split(  # pylint: disable=attribute-defined-outside-init
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self, augmentation: Optional[Transform] = transforms.ToTensor()) -> data.DataLoader:
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

    def val_dataloader(self) -> data.DataLoader:
        r"""
        Returns a dataloader for the validation subset of the dataset.
        """

        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> data.DataLoader:
        r"""
        Returns a dataloader for the testing subset of the dataset.
        """

        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def loss_weights(self) -> torch.Tensor:
        r"""
        Returns a tensor of weights for the different classes of the dataset to balance training.
        """

        class_counts = torch.bincount(torch.tensor(self.dataset.targets))

        return torch.tensor(1 / class_counts)
