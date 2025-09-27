"""Creates datasets using the Treespec pipeline."""

import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path

from treespec.dataset_creation.conf.config import DatasetCreationConfig
from treespec.dataset_creation.functions.create_dataset import (
    create_dataset,
    create_simple_dataset,
    create_big_scale_dataset,
)

cs = ConfigStore.instance()
cs.store(name="dataset_creation_config", node=DatasetCreationConfig)


@hydra.main(config_path="../conf", config_name="config")
def create_dataset_script(cfg: DatasetCreationConfig):
    """Script for creating a dataset from one run."""
    create_dataset(
        input_color_images_dir_path=Path(cfg.create_dataset.input_color_images_dir_path),
        input_color_image_filetype=cfg.create_dataset.input_color_image_filetype,
        input_color_images_format=cfg.create_dataset.input_color_images_format,
        input_segmentid_images_dir_path=Path(cfg.create_dataset.input_segmentid_images_dir_path),
        input_segmentid_image_filetype=cfg.create_dataset.input_segmentid_image_filetype,
        input_semanticclass_images_dir_path=Path(cfg.create_dataset.input_semanticclass_images_dir_path),
        input_semanticclass_image_filetype=cfg.create_dataset.input_semanticclass_image_filetype,
        pre_processed=cfg.create_dataset.pre_processed,
        date=cfg.create_dataset.date,
        run_number=cfg.create_dataset.run_number,
        processed_color_images_path=Path(cfg.create_dataset.processed_color_images_path),
        processed_color_image_filetype=cfg.create_dataset.processed_color_image_filetype,
        processed_segmentid_images_path=Path(cfg.create_dataset.processed_segmentid_images_path),
        processed_segmentid_image_filetype=cfg.create_dataset.processed_segmentid_image_filetype,
        processed_semanticclass_images_path=Path(cfg.create_dataset.processed_semanticclass_images_path),
        processed_semanticclass_image_filetype=cfg.create_dataset.processed_semanticclass_image_filetype,
        output_dataset_dir_path=Path(cfg.create_dataset.output_dataset_dir_path),
        input_tree_inventory_path=Path(cfg.create_dataset.input_tree_inventory_path),
        tree_attributes=cfg.create_dataset.tree_attributes,
    )


@hydra.main(config_path="../conf", config_name="config")
def create_simple_dataset_script(cfg: DatasetCreationConfig):
    """Script for creating a simple dataset from multiple runs."""
    create_simple_dataset(
        input_color_images_format=cfg.create_simple_dataset.input_color_images_format,
        date=cfg.create_simple_dataset.date,
        groundtruth_tree_inventory_path=Path(cfg.create_simple_dataset.groundtruth_tree_inventory_path),
        input_dir_path=Path(cfg.create_simple_dataset.input_dir_path),
        processed_dir_path=Path(cfg.create_simple_dataset.processed_dir_path),
        pre_processed=cfg.create_simple_dataset.pre_processed,
        output_dataset_dir_path=Path(cfg.create_simple_dataset.output_dataset_dir_path),
        run_numbers=cfg.create_simple_dataset.run_numbers,
    )


@hydra.main(config_path="../conf", config_name="config")
def create_big_scale_dataset_script(cfg: DatasetCreationConfig):
    """Script for creating a large-scale dataset."""
    create_big_scale_dataset(
        input_dir_path=Path(cfg.create_large_scale_dataset.input_dir_path),
        output_dir_path=Path(cfg.create_large_scale_dataset.output_dir_path),
        dates_and_runs=cfg.create_large_scale_dataset.dates_and_runs,
    )


if __name__ == "__main__":
    # Uncomment the desired function to run
    # create_dataset_script()  # pylint: disable=no-value-for-parameter
    # create_simple_dataset_script()  # pylint: disable=no-value-for-parameter
    create_big_scale_dataset_script()  # pylint: disable=no-value-for-parameter
    pass