from treespec.utils.matching_tools import match_predicted_tree_species
from treespec.conf.config import TreespecConfig
from treespec.conf.config_parser import train_config_values, predict_essen_config_values
from treespec.models.classification_model import ClassificationModel

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def predict_species(cfg: TreespecConfig):
    """Match predicted tree species from images to the matched inventory shapefile and writes it to the input path."""

    classification_model = ClassificationModel(
        model=train_config_values("model", cfg),
        model_weights=train_config_values("model_weights", cfg),
        num_classes=train_config_values("num_classes", cfg),
        loss_function=train_config_values("loss_function", cfg)(),
        learning_rate=train_config_values("learning_rate", cfg),
    )

    dataset = train_config_values("dataset", cfg)(
        data_dir=train_config_values("dataset_dir", cfg),
        batch_size=train_config_values("batch_size", cfg),
        num_workers=train_config_values("num_workers", cfg),
        use_ids=train_config_values("use_ids", cfg),
    )

    match_predicted_tree_species(
        classification_model=classification_model,
        dataset=dataset,
        tree_images_dir=predict_essen_config_values("tree_images_dir", cfg),
        input_inventory_path=predict_essen_config_values("input_inventory_path", cfg),
        output_inventory_path=predict_essen_config_values("output_inventory_path", cfg),
        trained_model_path=predict_essen_config_values("trained_model_path", cfg),
    )


if __name__ == "__main__":
    predict_species()
