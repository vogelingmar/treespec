"""Script to predict tree species from images and match them to an inventory shapefile."""

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
        model=predict_config_values("model", cfg),
        model_weights=predict_config_values("model_weights", cfg),
        dataset_dir_path=predict_config_values("dataset_dir", cfg),
        tree_images_dir_path=predict_config_values("tree_images_dir", cfg),
        input_inventory_path=predict_config_values("input_inventory_path", cfg),
        output_inventory_path=predict_config_values("output_inventory_path", cfg),
        trained_model_path=predict_config_values("trained_model_path", cfg),
    )


if __name__ == "__main__":
    predict()
