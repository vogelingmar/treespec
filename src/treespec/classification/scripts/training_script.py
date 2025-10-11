"""Trains a classification model on the image dataset."""

import pytorch_lightning as L
import hydra
from hydra.core.config_store import ConfigStore

from treespec.classification.conf.config import ClassificationConfig
from treespec.classification.conf.config_parser import train_config_values
from treespec.classification.functions.train import train as train_function

cs = ConfigStore.instance()
cs.store(name="classification_config", node=ClassificationConfig)


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: ClassificationConfig):
    """Training script for treespec Classification Pipeline"""

    train_function(
        model=train_config_values("model", cfg),
        model_weights=train_config_values("model_weights", cfg),
        input_dataset=train_config_values("dataset", cfg),
        dataset_dir_path=train_config_values("dataset_dir", cfg),
        num_classes=train_config_values("num_classes", cfg),
        use_ids=train_config_values("use_ids", cfg),
        epoch_count=train_config_values("epoch_count", cfg),
        batch_size=train_config_values("batch_size", cfg),
        num_workers=train_config_values("num_workers", cfg),
        learning_rate=train_config_values("learning_rate", cfg),
        input_loss_function=train_config_values("loss_function", cfg),
        trained_model_dir=train_config_values("trained_model_dir", cfg),
        train_augmentations=train_config_values("train_augmentations", cfg),
        pre_trained=train_config_values("pre_trained", cfg),
    )


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
