""" Sauen Dataset """

import torch
from torch.utils import data
from torchvision import datasets
import pytorch_lightning as L
from torchvision.transforms import v2

class Sauen_Dataset(L.LightningDataModule):
    
    def setup(self, transforms, batch_size):
        self.batch_size = batch_size

        train_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224)), 
            v2.RandomHorizontalFlip(0.2), 
            v2.RandomVerticalFlip(0.4), 
            v2.RandomRotation(15), 
            v2.RandomPerspective(distortion_scale=0.3), 
            v2.ColorJitter(),
            transforms
            ])

        self.dataset = datasets.ImageFolder(
            '/home/ingmar/Documents/repos/treespec/src/io/datasets/sauen_big/',
            transform=train_transforms
        )
         
        total_size = len(self.dataset)
        val_size = int(0.1 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size
        self.train, self.val, self.test = data.random_split(
            self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )
    
    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size)
    
    def loss_weights(self):
        class_counts = torch.bincount(torch.tensor(self.dataset.targets))
        return torch.tensor(1/class_counts)
    
    
