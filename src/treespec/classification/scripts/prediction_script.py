"""Script to predict tree species from images and match them to an inventory shapefile."""

# TODO: refactor script to classification and work
from treespec.classification.conf.config import ClassificationConfig
from treespec.classification.conf.config_parser import train_config_values, predict_config_values
from treespec.classification.functions.predict import predict as predict_function

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="classification_config", node=ClassificationConfig)


@hydra.main(config_path="../conf", config_name="config")
def predict(cfg: ClassificationConfig):
    """Match predicted tree species from images to the matched inventory shapefile and writes it to the output path."""

    predict_function(
        model=train_config_values("model", cfg),
        model_weights=train_config_values("model_weights", cfg),
        num_classes=train_config_values("num_classes", cfg),
        loss_function=train_config_values("loss_function", cfg),
        learning_rate=train_config_values("learning_rate", cfg),
        dataset=train_config_values("dataset", cfg),
        dataset_dir=train_config_values("dataset_dir", cfg),
        batch_size=train_config_values("batch_size", cfg),
        num_workers=train_config_values("num_workers", cfg),
        use_ids=train_config_values("use_ids", cfg),
        tree_images_dir=predict_config_values("tree_images_dir", cfg),
        input_inventory_path=predict_config_values("input_inventory_path", cfg),
        output_inventory_path=predict_config_values("output_inventory_path", cfg),
        trained_model_path=predict_config_values("trained_model_path", cfg),
    )


if __name__ == "__main__":
    predict()
