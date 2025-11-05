import pytest
import os
from types import SimpleNamespace
from pathlib import Path
from treespec.classification.conf import config_parser


classification_mock_temp_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock", "temp")

@pytest.fixture
def mock_classification_config():
    return SimpleNamespace(
        train=SimpleNamespace(
            model="resnet50",
            model_weights="resnet50_default",
            dataset="folder",
            dataset_dir_path=Path("/test_path/mock"),
            num_classes=7,
            use_ids=True,
            epoch_count=20,
            batch_size=5,
            num_workers=27,
            learning_rate=0.001,
            loss_function="cross_entropy",
            trained_model_dir_path=Path(classification_mock_temp_dir_path),
            trained_model_path=Path(os.path.join(classification_mock_temp_dir_path, "mock_trained_model.pth")),
            train_augmentations=[
                {"name": "RandomVerticalFlip", "p": 0.1},
                {"name": "RandomHorizontalFlip", "p": 0.5},
                {"name": "RandomRotation", "degrees": 15},
                {"name": "ColorJitter", "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
                {"name": "RandomResizedCrop", "size": [224, 224]},
                {"name": "RandomPerspective", "distortion_scale": 0.3, "p": 0.3},
                {"name": "ElasticTransform", "alpha": 20, "sigma": 20},
            ],
        ),
        predict=SimpleNamespace(
            tree_images_dir_path=Path("/test_path/mock"),
            input_inventory_path=Path("/test_path/mock"),
            output_inventory_path=Path("/test_path/mock"),
            trained_model_path=Path("/test_path/mock"),
        ),
    )


def test_train_config_values(mock_classification_config):
    assert callable(config_parser.train_config_values("model", mock_classification_config))
    assert hasattr(config_parser.train_config_values("model_weights", mock_classification_config), "transforms")
    assert callable(config_parser.train_config_values("dataset", mock_classification_config))
    assert callable(config_parser.train_config_values("loss_function", mock_classification_config))
    assert hasattr(config_parser.train_config_values("train_augmentations", mock_classification_config), "__call__")
    assert isinstance(config_parser.train_config_values("dataset_dir_path", mock_classification_config), Path)
    assert isinstance(config_parser.train_config_values("num_classes", mock_classification_config), int)
    assert isinstance(config_parser.train_config_values("use_ids", mock_classification_config), bool)
    assert isinstance(config_parser.train_config_values("epoch_count", mock_classification_config), int)
    assert isinstance(config_parser.train_config_values("batch_size", mock_classification_config), int)
    assert isinstance(config_parser.train_config_values("num_workers", mock_classification_config), int)
    assert isinstance(config_parser.train_config_values("learning_rate", mock_classification_config), float)
    assert isinstance(config_parser.train_config_values("trained_model_dir_path", mock_classification_config), Path)
    assert isinstance(config_parser.train_config_values("trained_model_path", mock_classification_config), Path)


def test_predict_config_values(mock_classification_config):
    assert callable(config_parser.train_config_values("model", mock_classification_config))
    assert hasattr(config_parser.train_config_values("model_weights", mock_classification_config), "transforms")
    assert isinstance(config_parser.train_config_values("dataset_dir_path", mock_classification_config), Path)
    assert isinstance(config_parser.predict_config_values("tree_images_dir_path", mock_classification_config), Path)
    assert isinstance(config_parser.predict_config_values("input_inventory_path", mock_classification_config), Path)
    assert isinstance(config_parser.predict_config_values("output_inventory_path", mock_classification_config), Path)
    assert isinstance(config_parser.predict_config_values("trained_model_path", mock_classification_config), Path)
