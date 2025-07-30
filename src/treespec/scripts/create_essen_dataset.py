"""Create the Essen dataset from the RGB and SegmentID Images."""

import os
import shutil
import hydra
from hydra.core.config_store import ConfigStore

from treespec.utils import image_tools
from treespec.utils.matching_tools import create_dictionary

from treespec.conf.config_parser import create_essen_dataset_config_values as config_values
from treespec.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):
    """Script that creates the Essen dataset from RGB, SegmentID and SemanticClass images and inventory data."""

    if config_values("apply_center_crop", cfg):
        image_tools.select_rgb_images(
            input_dir=config_values["original_color_images_path"],
            output_dir=config_values["color_images_path"],
            image_type=config_values["color_type"],
        )

    tree_attributes_dict = create_dictionary(config_values("attribute_path", cfg))

    if not config_values("pictures_extracted", cfg):

        image_tools.extract_pano_faces(
            input_dir=config_values("original_color_images_path", cfg),
            output_dir=config_values("color_images_path", cfg),
            input_file_type=config_values("color_type", cfg),
            output_file_type=config_values("color_output_type", cfg),
            run_number=config_values("run", cfg),
            apply_center_crop=config_values("apply_center_crop", cfg),
        )

        image_tools.extract_pano_faces(
            input_dir=config_values("original_id_images_path", cfg),
            output_dir=config_values("segmentid_images_path", cfg),
            input_file_type=config_values("seg_type", cfg),
            output_file_type=config_values("seg_output_type", cfg),
            run_number=config_values("run", cfg),
            filter=config_values("filter_id", cfg),
            apply_center_crop=config_values("apply_center_crop", cfg),
        )

        image_tools.extract_pano_faces(
            input_dir=config_values("original_sem_images_path", cfg),
            output_dir=config_values("semantic_images_path", cfg),
            input_file_type=config_values("sem_type", cfg),
            output_file_type=config_values("sem_output_type", cfg),
            run_number=config_values("run", cfg),
            filter=config_values("filter_semantic", cfg),
            apply_center_crop=config_values("apply_center_crop", cfg),
        )

    image_tools.find_all_trees(
        segmentid_dir=config_values("segmentid_images_path", cfg),
        color_dir=config_values("color_images_path", cfg),
        output_dir=config_values("output_trees_dir", cfg),
        tree_attributes_dict=tree_attributes_dict,
        semantic_dir=config_values("semantic_images_path", cfg),
        cover=config_values("mask", cfg),
        input_file_type="png",
    )
    output_trees_dir = config_values("output_trees_dir", cfg)

    image_tools.create_dataset(output_trees_dir, output_trees_dir, only_copy=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
