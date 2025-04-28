"""Test for SauenDataset datamodule"""

# pylint: disable=redefined-outer-name
import os
import pytest

from treespec.datasets.image_dataset import ImageDataset


@pytest.fixture
def sauen_dataset():
    """Fixture that holds a SauenDataset instance"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mock/sauen_v1")
    return ImageDataset(data_dir=data_dir, batch_size=5, num_workers=27)


def test_setup(sauen_dataset):
    """Tests the setup method of SauenDataset"""
    sauen_dataset.setup()
    assert len(sauen_dataset.dataset) > 0
    assert len(sauen_dataset.train) > 0
    assert len(sauen_dataset.val) > 0
    assert len(sauen_dataset.test) > 0


def test_dataloaders(sauen_dataset):
    """Tests the dataloaders of SauenDataset"""
    sauen_dataset.setup()
    train_loader = sauen_dataset.train_dataloader()
    val_loader = sauen_dataset.val_dataloader()
    test_loader = sauen_dataset.test_dataloader()

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_loss_weights(sauen_dataset):
    """Tests the loss weights of SauenDataset"""
    sauen_dataset.setup()
    loss_weights = sauen_dataset.loss_weights()
    assert len(loss_weights) > 0
