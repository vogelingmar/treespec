"""Test for Image Dataset datamodule"""

# pylint: disable=redefined-outer-name
import os
import pytest
import torch

from treespec.classification.image_dataset import ImageDataset


@pytest.fixture
def sorted_dataset():
    """Fixture that holds a sorted dataset instance"""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset_creation", "mock", "datasets", "dataset_sorted"
    )
    return ImageDataset(dataset_dir_path=data_dir, batch_size=5, num_workers=27, use_ids=False)


@pytest.mark.parametrize("use_ids", [True, False])
def test_setup(use_ids):
    """Tests the setup method of ImageDataset and checks for dataset leakage when use_ids is True"""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset_creation", "mock", "datasets", "dataset_sorted"
    )
    faulty_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset_creation", "mock", "datasets", "dataset_sorted_small"
    )
    dataset = ImageDataset(dataset_dir_path=data_dir, batch_size=5, num_workers=2, use_ids=use_ids)
    faulty_dataset = ImageDataset(dataset_dir_path=faulty_data_dir, batch_size=5, num_workers=2, use_ids=use_ids)
    dataset.setup()

    with pytest.raises(ValueError):
        faulty_dataset.setup()

    assert len(dataset.dataset) > 0
    assert len(dataset.train) > 0
    assert len(dataset.val) > 0
    assert len(dataset.test) > 0

    # For Subset, need to access indices and underlying dataset
    def extract_ids(subset):
        if hasattr(subset, "indices"):
            return set(os.path.basename(subset.dataset.samples[i][0]).split("_")[0] for i in subset.indices)
        else:
            # fallback for ImageFolder
            return set(os.path.basename(sample[0]).split("_")[0] for sample in subset.samples)

    train_ids = extract_ids(dataset.train)
    val_ids = extract_ids(dataset.val)
    test_ids = extract_ids(dataset.test)

    if use_ids:
        # Assert no overlap between splits
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
    # else: no assertion, overlap is possible


def test_dataloaders(sorted_dataset):
    """Tests the dataloaders of Image Dataset"""
    sorted_dataset.setup()
    train_loader = sorted_dataset.train_dataloader()
    val_loader = sorted_dataset.val_dataloader()
    test_loader = sorted_dataset.test_dataloader()

    assert type(train_loader) == type(val_loader) == type(test_loader)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_loss_weights(sorted_dataset):
    """Tests the loss weights of Image Dataset"""
    sorted_dataset.setup()
    loss_weights = sorted_dataset.loss_weights()
    assert len(loss_weights) > 0
