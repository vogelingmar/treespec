"""Test for the classification model"""

# pylint: disable=redefined-outer-name
import os
import numpy as np
import pytest
import torch
import pytorch_lightning as L
from torchvision.models import (  # type: ignore
    resnet50,
    ResNet50_Weights,
)
from treespec.classification.classification_model import ClassificationModel
from treespec.classification.image_dataset import ImageDataset


@pytest.fixture
def classification_model():
    """ClassificationModel instance for testing"""
    return ClassificationModel(
        model_weights=ResNet50_Weights.DEFAULT,
        model=resnet50,
        num_classes=5,
        loss_function=torch.nn.CrossEntropyLoss(),
        learning_rate=0.001,
    )


def test_forward(classification_model):
    """Tests the forward pass of the ClassificationModel"""
    inputs = torch.randn(1, 3, 224, 224)
    outputs = classification_model(inputs)
    assert outputs.shape == (1, 5)


def test_training_step(classification_model):
    """Tests the training step of the ClassificationModel"""
    batch = [torch.randn(1, 3, 224, 224), torch.tensor([0])]
    loss = classification_model.training_step(batch, 0)
    assert loss.item() > 0


def test_predict(classification_model):
    """Tests the predict method of the ClassificationModel"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image = os.path.join(
        base_dir, "dataset_creation/mock/datasets/dataset_unsorted/tree_crop/36_date_run_rand_23left_Amb.png"
    )
    prediction = classification_model.predict(image)
    assert 0 <= prediction["category"] <= 4
    assert prediction["score"] > 0


def test_configure_optimizers(classification_model):
    """Tests the configure_optimizers method of the ClassificationModel"""
    optimizer = classification_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


def test_overfitting(classification_model):
    """Tries to overfit one training batch in order to test that the model is able to fit the training data."""
    trainer = L.Trainer(max_epochs=15, log_every_n_steps=30, deterministic=True, benchmark=True)

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset_creation", "mock", "datasets", "dataset_sorted"
    )
    dataset = ImageDataset(dataset_dir_path=data_dir, batch_size=5, num_workers=8, use_ids=False)
    dataset.setup()

    default_transforms = ResNet50_Weights.DEFAULT.transforms()

    train_batch = next(iter(dataset.train_dataloader(default_transforms)))  # pylint: disable=too-many-function-args
    untrained_batch = next(iter(dataset.test_dataloader()))
    initial_loss = classification_model._common_steps(  # pylint: disable=protected-access
        train_batch, 0, "test", False
    ).item()  # pylint: disable=protected-access

    trainer.fit(
        model=classification_model,
        train_dataloaders=dataset.train_dataloader(default_transforms),  # pylint: disable=too-many-function-args
    )  # pylint: disable=too-many-function-args
    final_loss = classification_model._common_steps(  # pylint: disable=protected-access
        train_batch, 0, "test", False
    ).item()  # pylint: disable=protected-access

    assert final_loss < initial_loss
    assert final_loss < 0.4

    untrained_loss = classification_model._common_steps(  # pylint: disable=protected-access
        untrained_batch, 0, "test", False
    ).item()  # pylint: disable=protected-access
    assert untrained_loss > final_loss + 0.3


def test_calculate_per_class_metrics(classification_model):
    """Tests the calculate_per_class_metrics method of the ClassificationModel"""
    predictions = torch.tensor([0, 1, 2, 1])
    labels = torch.tensor([0, 1, 1, 2])
    metrics = classification_model.calculate_per_class_metrics(predictions, labels)
    assert set(metrics.keys()) == {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "f1_score",
    }
    assert all(isinstance(v, torch.Tensor) for v in metrics.values())


def test_predict_step(classification_model):
    """Tests the predict_step method of the ClassificationModel"""
    batch = torch.randn(1, 3, 224, 224)
    class_id, score = classification_model.predict_step(batch, 0)
    assert isinstance(class_id, int)
    assert isinstance(score, float)
