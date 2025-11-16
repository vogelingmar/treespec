"""Trains a classification model on the image dataset."""

import hydra
from hydra.core.config_store import ConfigStore

from treespec.classification.conf.config import ClassificationConfig
from treespec.classification.conf.config_parser import train_config_values
from treespec.classification.functions.train import train as train_function

cs = ConfigStore.instance()
cs.store(name="classification_config", node=ClassificationConfig)


@hydra.main(config_path="../conf", config_name="config_model_comparison")
def train(cfg: ClassificationConfig):  # pylint: disable=too-many-branches
    """Training script for treespec Classification Pipeline"""

    if (model := train_config_values("model", cfg)) is None:
        raise ValueError("Model not specified in configuration.")

    if (model_weights := train_config_values("model_weights", cfg)) is None:
        raise ValueError("Model weights not specified in configuration.")

    if (dataset_dir_path := train_config_values("dataset_dir_path", cfg)) is None:
        raise ValueError("Dataset directory not specified in configuration.")

    if (num_classes := train_config_values("num_classes", cfg)) is None:
        raise ValueError("Number of classes not specified in configuration.")

    if (use_ids := train_config_values("use_ids", cfg)) is None:
        raise ValueError("Use IDs flag not specified in configuration.")

    if (epoch_count := train_config_values("epoch_count", cfg)) is None:
        raise ValueError("Epoch count not specified in configuration.")

    if (batch_size := train_config_values("batch_size", cfg)) is None:
        raise ValueError("Batch size not specified in configuration.")

    if (num_workers := train_config_values("num_workers", cfg)) is None:
        raise ValueError("Number of workers not specified in configuration.")

    if (learning_rate := train_config_values("learning_rate", cfg)) is None:
        raise ValueError("Learning rate not specified in configuration.")

    if (input_loss_function := train_config_values("loss_function", cfg)) is None:
        raise ValueError("Loss function not specified in configuration.")

    if (trained_model_dir_path := train_config_values("trained_model_dir", cfg)) is None:
        raise ValueError("Trained model directory path not specified in configuration.")

    if (trained_model_path := train_config_values("trained_model_path", cfg)) is None:
        raise ValueError("Trained model path not specified in configuration.")

    if (train_augmentations := train_config_values("train_augmentations", cfg)) is None:
        raise ValueError("Train augmentations not specified in configuration.")

    train_function(
        model=model,
        model_weights=model_weights,
        dataset_dir_path=dataset_dir_path,
        num_classes=num_classes,
        use_ids=use_ids,
        epoch_count=epoch_count,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        input_loss_function=input_loss_function,
        trained_model_dir_path=trained_model_dir_path,
        trained_model_path=trained_model_path,
        train_augmentations=train_augmentations,
    )


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
