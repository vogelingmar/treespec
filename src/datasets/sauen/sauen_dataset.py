""" Sauen Dataset """

import torch
from torch.utils import data
from torchvision import datasets
import pytorch_lightning as L
from torchvision.transforms import v2

class Sauen_Dataset(L.LightningDataModule):

    def prepare_data(self, transforms):

        self.dataset = datasets.ImageFolder(
            '/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2',
            transform=transforms
        )

    def setup(self):

        total_size = len(self.dataset)
        val_size = int(0.1 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size
        self.train, self.val, self.test = data.random_split(
            self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self, augmentations, batch_size: int = 5):
        self.train.dataset(transform=augmentations)
        return data.DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=27)

    def val_dataloader(self, batch_size: int = 5):
        return data.DataLoader(self.val, batch_size=batch_size)
    
    def test_dataloader(self, batch_size: int = 5):
        return data.DataLoader(self.test, batch_size=batch_size)
    
    def loss_weights(self):
        class_counts = torch.bincount(torch.tensor(self.dataset.targets))
        return torch.tensor(1/class_counts)