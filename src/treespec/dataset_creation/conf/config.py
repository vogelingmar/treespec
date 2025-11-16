"""Definition of the dataset creation config parameters"""

# pylint: disable=too-many-instance-attributes

from pathlib import Path

from dataclasses import dataclass


@dataclass
class DatasetCreationParams:
    """Configuration of parameters for the create_essen_dataset script.

    Note:
    Give the path of the input_tree_inventory_path WITHOUT the .shp ending."""

    input_color_images_dir_path: Path
    input_color_image_filetype: str
    input_color_images_format: str
    input_segmentid_images_dir_path: Path
    input_segmentid_image_filetype: str
    input_semanticclass_images_dir_path: Path
    input_semanticclass_image_filetype: str
    pre_processed: bool
    date: str
    run_number: int
    processed_color_images_path: Path
    processed_color_image_filetype: str
    processed_segmentid_images_path: Path
    processed_segmentid_image_filetype: str
    processed_semanticclass_images_path: Path
    processed_semanticclass_image_filetype: str
    output_dataset_dir_path: Path
    input_tree_inventory_path: Path
    tree_attributes: list


@dataclass
class SimpleDatasetCreationParams:
    """Configuration of parameters for the create_essen_dataset script.

    Note:
    Give the path of the groundtruth_tree_inventory_path WITHOUT the .shp ending."""

    input_color_images_format: str
    groundtruth_tree_inventory_path: Path
    input_dir_path: Path
    processed_dir_path: Path
    pre_processed: bool
    output_dataset_dir_path: Path
    date: str
    run_numbers: list


@dataclass
class LargeScaleDatasetCreationParams:
    """Configuration of parameters for the create_essen_dataset_big script."""

    input_dir_path: Path
    output_dir_path: Path
    dates_and_runs: dict


@dataclass
class TreeInventoryMatchingParams:
    """Configuration of parameters for the matching script.

    Note:
    Give the path of the inventories WITHOUT the .shp ending."""

    predicted_tree_inventory_path: Path
    groundtruth_tree_inventory_path: Path
    matched_tree_inventory_output_path: Path
    use_dbh_matching_filter: bool


@dataclass
class DatasetCreationConfig:
    """Configuration of the configs going into the treespec config."""

    create_dataset: DatasetCreationParams
    create_simple_dataset: SimpleDatasetCreationParams
    create_large_scale_dataset: LargeScaleDatasetCreationParams
    match_inventories: TreeInventoryMatchingParams
