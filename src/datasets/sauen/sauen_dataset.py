"""Sauen Dataset"""

import torch
from torch.utils import data
from torchvision import datasets
import pytorch_lightning as L
from torchvision.transforms import v2


class Sauen_Dataset(L.LightningDataModule):

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

        # add download path for the images of the dataset
        pass

    def setup(self, transforms):

        # create image folder dataset from images
        self.dataset = datasets.ImageFolder(self.data_dir, transform=transforms)

        # calculation of different set sizes (80% traning, 10% validation, 10% test)
        total_size = len(self.dataset)
        val_size = int(0.1 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size

        self.train, self.val, self.test = data.random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        # , generator=torch.Generator().manual_seed(42): could be added as parameter to random_split to seed randomness

    def train_dataloader(self):

        # self.train.dataset(transform=augmentations)
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):

        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):

        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):

        # return super().predict_dataloader()
        pass

    def loss_weights(self):

        class_counts = torch.bincount(torch.tensor(self.dataset.targets))
        return torch.tensor(1 / class_counts)
