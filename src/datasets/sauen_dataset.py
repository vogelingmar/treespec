""" Sauen Dataset """

import torch
from torch.utils import data
from torchvision import datasets
import pytorch_lightning as L

class Sauen_Dataset(L.LightningDataModule):
    
    def setup(self, transforms):

            dataset = datasets.ImageFolder(
                '/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen/',
                transform=transforms
            )

            total_size = len(dataset)

            val_size = int(0.1 * total_size)
            test_size = int(0.1 * total_size)
            train_size = total_size - val_size - test_size

            self.train, self.val, self.test = data.random_split(
                dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
            )
    
    def train_dataloader(self):
        return data.DataLoader(self.train)

    def val_dataloader(self):
        return data.DataLoader(self.val)
    
    def test_dataloader(self):
        return data.DataLoader(self.test)
    
