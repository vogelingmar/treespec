import torch
from torchvision import datasets

import pytest

from src.datasets.sauen.sauen_dataset import SauenDataset


def test_setup():

    sd = SauenDataset()  # uses default values
    sd.setup()

    assert type(sd.dataset) == datasets.ImageFolder
    assert sd.train is not None
    assert sd.val is not None
    assert sd.test is not None
