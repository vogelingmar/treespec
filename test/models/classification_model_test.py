import os
import pytest
import torch
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision.io import decode_image

from src.treespec.models.classification_model import ClassificationModel

@pytest.fixture
def classification_model():
    return ClassificationModel(
        model_weights=ResNet50_Weights.DEFAULT,
        model=resnet50,
        num_classes=3,
        loss_function=torch.nn.CrossEntropyLoss(),
        learning_rate=0.001,
    )

def test_init_error():
    with pytest.raises(AttributeError):
        ClassificationModel(model_weights=EfficientNet_V2_L_Weights.DEFAULT, model=efficientnet_v2_l)

def test_forward(classification_model):
    inputs = torch.randn(1, 3, 224, 224)
    outputs = classification_model(inputs)
    assert outputs.shape == (1, 3)

def test_training_step(classification_model):
    batch = [torch.randn(1, 3, 224, 224), torch.tensor([0])]
    loss = classification_model.training_step(batch, 0)
    assert loss.item() > 0

def test_predict(classification_model):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image = os.path.join(base_dir, "mock/sauen_v1/beech/beech_test.jpg")
    prediction = classification_model.predict(image)
    assert 0 <= prediction["category"] <= 2 
    assert prediction["score"] > 0

def test_configure_optimizers(classification_model):
    optimizer = classification_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)