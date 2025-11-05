import os
import shutil

from torch import nn
import torchvision.transforms.v2 as transforms

from torchvision.models import googlenet, GoogLeNet_Weights
from treespec.classification.image_dataset import ImageDataset

from treespec.classification.functions import train


def test_train():
    mock_temp_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock", "temp")
    shutil.rmtree(mock_temp_dir_path, ignore_errors=True)

    mock_dataset_creation_dir_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset_creation", "mock"
    )

    train.train(
        model=googlenet,
        model_weights=GoogLeNet_Weights.DEFAULT,
        input_dataset=ImageDataset,
        dataset_dir_path=os.path.join(mock_dataset_creation_dir_path, "datasets", "dataset_sorted"),
        num_classes=5,
        use_ids=True,
        epoch_count=2,
        batch_size=3,
        num_workers=0,
        learning_rate=0.001,
        input_loss_function=nn.CrossEntropyLoss,
        trained_model_dir_path=os.path.join(mock_temp_dir_path, "trained_model"),
        train_augmentations=transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop(224)])
    )

    assert len(os.listdir(mock_temp_dir_path)) > 0
    shutil.rmtree(mock_temp_dir_path, ignore_errors=True)