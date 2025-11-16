"""Script to predict tree species from images and match them to an inventory shapefile."""

import hydra
from hydra.core.config_store import ConfigStore
from treespec.classification.conf.config import ClassificationConfig
from treespec.classification.conf.config_parser import predict_config_values
from treespec.classification.functions.predict import predict_species


cs = ConfigStore.instance()
cs.store(name="classification_config", node=ClassificationConfig)


@hydra.main(config_path="../conf", config_name="config")
def predict(cfg: ClassificationConfig):
    """Match predicted tree species from images to the matched inventory shapefile and writes it to the output path."""

    if (model := predict_config_values("model", cfg)) is None:
        raise ValueError("Model not specified in configuration.")

    if (model_weights := predict_config_values("model_weights", cfg)) is None:
        raise ValueError("Model weights not specified in configuration.")

    if (dataset_dir_path := predict_config_values("dataset_dir_path", cfg)) is None:
        raise ValueError("Dataset directory path not specified in configuration.")

    if (tree_images_dir_path := predict_config_values("tree_images_dir_path", cfg)) is None:
        raise ValueError("Tree images directory path not specified in configuration.")

    if (input_inventory_path := predict_config_values("input_inventory_path", cfg)) is None:
        raise ValueError("Input inventory path not specified in configuration.")

    if (output_inventory_path := predict_config_values("output_inventory_path", cfg)) is None:
        raise ValueError("Output inventory path not specified in configuration.")

    if (trained_model_path := predict_config_values("trained_model_path", cfg)) is None:
        raise ValueError("Trained model path not specified in configuration.")

    predict_species(
        model=model,
        model_weights=model_weights,
        dataset_dir_path=dataset_dir_path,
        tree_images_dir_path=tree_images_dir_path,
        input_inventory_path=input_inventory_path,
        output_inventory_path=output_inventory_path,
        trained_model_path=trained_model_path,
    )


if __name__ == "__main__":
    predict()  # pylint: disable=no-value-for-parameter
