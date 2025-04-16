import pytest
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import decode_image

from src.models.classification_model import ClassificationModel


def test_forward():
    classification_model = ClassificationModel(
        model_weights=ResNet50_Weights.DEFAULT, num_classes=3
    )
    img_path = "/home/ingmar/Documents/repos/treespec/src/datasets/sauen/images/sauen_v2/beech/bark_4068_box_00_angle_-5.34.jpg"
    picture = decode_image(img_path)
    transforms = ResNet50_Weights.DEFAULT.transforms()
    batch = transforms(picture).unsqueeze(0)
    prediction = classification_model(batch)

    assert prediction.shape == (1, 3)


def test_calculate_metrics():
    classification_model = ClassificationModel(
        model_weights=ResNet50_Weights.DEFAULT, num_classes=3
    )
    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    labels = torch.tensor([2, 2])
    accuracy, f1, precision, recall = classification_model.calculate_metrics(outputs, labels)
    assert accuracy.item() == 0.5
    assert f1.item() == 0.5
    assert precision.item() == 0.5
    assert recall.item() == 0.5


def test_training():
    classification_model = ClassificationModel(
        model_weights=ResNet50_Weights.DEFAULT, num_classes=3
    )
    outputs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]])
    labels = torch.tensor([2, 2, 2])
    batch = [outputs, labels]
    loss = classification_model.training_step(batch, 0)
    assert loss.item() == 50
