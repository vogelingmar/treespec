"""Test for SauenDataset datamodule"""

# pylint: disable=redefined-outer-name
import os
import pytest

from treespec.datasets.image_dataset import ImageDataset


@pytest.fixture
def sauen_dataset():
    """Fixture that holds a SauenDataset instance"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mock/sauen_v1")
    return ImageDataset(data_dir=data_dir, batch_size=5, num_workers=27, use_ids=False)

@pytest.mark.parametrize("use_ids", [True, False])
def test_setup(use_ids):
    """Tests the setup method of SauenDataset and checks for dataset leakage when use_ids is True"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mock/essen_mock/run_70/big_dataset_70")
    dataset = ImageDataset(data_dir=data_dir, batch_size=5, num_workers=2, use_ids=use_ids)
    dataset.setup()

    assert len(dataset.dataset) > 0
    assert len(dataset.train) > 0
    assert len(dataset.val) > 0
    assert len(dataset.test) > 0

    def get_tree_ids(split):
        ids = set()
        for sample in split:
            # sample is (img, label) tuple for ImageFolder
            if hasattr(sample, 'path'):  # for Subset of ImageFolder
                path = sample.path
            elif isinstance(sample, tuple):
                # For Subset, sample[0] is PIL image, sample[1] is label, but we need the path
                # So we need to access the underlying dataset
                # This works for torch.utils.data.Subset
                idx = split.indices[split.dataset.index(sample)]
                path = split.dataset.dataset.samples[idx][0]
            else:
                continue
            filename = os.path.basename(path)
            tree_id = filename.split("_")[0]
            ids.add(tree_id)
        return ids

    # For Subset, need to access indices and underlying dataset
    def extract_ids(subset):
        if hasattr(subset, 'indices'):
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
