"""Test for the classification model"""

# pylint: disable=redefined-outer-name
import os
import pytest
import torch
import pytorch_lightning as L
from torchvision.models import (  # type: ignore
    resnet50,
    ResNet50_Weights,
)
from treespec.models.classification_model import ClassificationModel
from treespec.datasets.image_dataset import ImageDataset


@pytest.fixture
def classification_model():
    """Fixture that holds a ClassificationModel instance"""
    return ClassificationModel(
        model_weights=ResNet50_Weights.DEFAULT,
        model=resnet50,
        num_classes=3,
        loss_function=torch.nn.CrossEntropyLoss(),
        learning_rate=0.001,
    )


def test_forward(classification_model):
    """Tests the forward pass of the ClassificationModel"""
    inputs = torch.randn(1, 3, 224, 224)
    outputs = classification_model(inputs)
    assert outputs.shape == (1, 3)


def test_training_step(classification_model):
    """Tests the training step of the ClassificationModel"""
    batch = [torch.randn(1, 3, 224, 224), torch.tensor([0])]
    loss = classification_model.training_step(batch, 0)
    assert loss.item() > 0


def test_predict(classification_model):
    """Tests the predict method of the ClassificationModel"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image = os.path.join(base_dir, "mock/sauen_v1/beech/beech_test.jpg")
    prediction = classification_model.predict(image)
    assert 0 <= prediction["category"] <= 2
    assert prediction["score"] > 0


def test_configure_optimizers(classification_model):
    """Tests the configure_optimizers method of the ClassificationModel"""
    optimizer = classification_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


def test_overfitting(classification_model):
    """Tries to overfit one training batch in order to test that the model is able to fit the training data."""
    trainer = L.Trainer(max_epochs=15, log_every_n_steps=30)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mock/sauen_v1")
    sauen_dataset = ImageDataset(data_dir=data_dir, batch_size=5, num_workers=27)
    sauen_dataset.setup()

    default_transforms = ResNet50_Weights.DEFAULT.transforms()

    train_batch = next(
        iter(sauen_dataset.train_dataloader(default_transforms))  # pylint: disable=too-many-function-args
    )
    untrained_batch = next(iter(sauen_dataset.test_dataloader()))
    initial_loss = classification_model._common_steps(  # pylint: disable=protected-access
        train_batch, 0, "test", False
    ).item()  # pylint: disable=protected-access

    trainer.fit(
        model=classification_model,
        train_dataloaders=sauen_dataset.train_dataloader(default_transforms),  # pylint: disable=too-many-function-args
    )  # pylint: disable=too-many-function-args
    final_loss = classification_model._common_steps(  # pylint: disable=protected-access
        train_batch, 0, "test", False
    ).item()  # pylint: disable=protected-access

    assert final_loss < initial_loss
    assert final_loss < 0.4

    untrained_loss = classification_model._common_steps(  # pylint: disable=protected-access
        untrained_batch, 0, "test", False
    ).item()  # pylint: disable=protected-access
    assert untrained_loss > final_loss + 0.5
