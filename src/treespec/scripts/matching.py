"""Matching script for cadastre shapefile and predicted_cadastre shapefile."""

import hydra
from hydra.core.config_store import ConfigStore

from treespec.conf.config import TreespecConfig
from treespec.conf.config_parser import matching_config_values

from treespec.utils.matching_tools import match_and_export

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):
    """Matching manually maintained inventory with predicted inventory."""
    attributes_path = matching_config_values("predicted_cadastre_path", cfg)
    cadastre_path = matching_config_values("cadastre_path", cfg)
    output_path = matching_config_values("output_path", cfg)
    use_dbh_filter = matching_config_values("use_dbh_filter", cfg)
    match_and_export(attributes_path, cadastre_path, output_path, use_dbh_filter)


if __name__ == "__main__":
    main()
