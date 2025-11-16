"""Matches predicted and ground truth tree inventories."""

import hydra
from hydra.core.config_store import ConfigStore
from treespec.dataset_creation.conf.config import DatasetCreationConfig
from treespec.dataset_creation.functions.match_inventories import match

cs = ConfigStore.instance()
cs.store(name="dataset_creation_config", node=DatasetCreationConfig)


@hydra.main(config_path="../conf", config_name="config")
def match_inventories(cfg: DatasetCreationConfig):
    """Matches predicted and ground truth inventories and exports the result."""

    match(
        predicted_inventory_path=cfg.match_inventories.predicted_tree_inventory_path,
        groundtruth_inventory_path=cfg.match_inventories.groundtruth_tree_inventory_path,
        output_inventory_path=cfg.match_inventories.matched_tree_inventory_output_path,
        use_dbh_filter=cfg.match_inventories.use_dbh_matching_filter,
    )


if __name__ == "__main__":
    match_inventories()  # pylint: disable=no-value-for-parameter
