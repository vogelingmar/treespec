"""Type definitions for classification configuration values."""
from pathlib import Path

from dataclasses import dataclass


@dataclass
class TrainParams:  # pylint: disable=too-many-instance-attributes
    """Datatype defintion for config values of the deep learning model training script."""

    model: str
    model_weights: str
    dataset: str
    dataset_dir_path: Path
    num_classes: int
    use_ids: bool
    epoch_count: int
    batch_size: int
    num_workers: int
    learning_rate: float
    loss_function: str
    trained_model_dir_path: Path
    train_augmentations: list


@dataclass
class PredictParams:
    """Datatype definition for config values of the deep learning model prediction script.
    
    Note:
    Give the path for the inventories without the file extension."""

    tree_images_dir_path: Path
    input_inventory_path: Path
    output_inventory_path: Path
    trained_model_path: Path


@dataclass
class ClassificationConfig:
    """Subclass definition for the classification configuration."""

    train: TrainParams
    predict: PredictParams
