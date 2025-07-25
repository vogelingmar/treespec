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
    """Script that creates the Essen dataset from RGB and SegmentID images."""

    tree_attributes_dict = create_dictionary(config_values("attribute_path", cfg))
    # image_tools.select_rgb_images(
    #    input_dir=config_values["original_color_images_path"],
    #    output_dir=config_values["color_images_path"],
    #    image_type=config_values["color_type"],
    # )

    if not config_values("pictures_extracted", cfg):

        image_tools.extract_pano_faces(
            input_dir=config_values("original_color_images_path", cfg),
            output_dir=config_values("color_images_path", cfg),
            input_file_type=config_values("color_type", cfg),
            output_file_type=config_values("color_output_type", cfg),
            run_number=config_values("run", cfg),
            apply_center_crop=False,
        )

        image_tools.extract_pano_faces(
            input_dir=config_values("original_id_images_path", cfg),
            output_dir=config_values("segmentid_images_path", cfg),
            input_file_type=config_values("seg_type", cfg),
            output_file_type=config_values("seg_output_type", cfg),
            run_number=config_values("run", cfg),
            filter=config_values("filter_id", cfg),
            apply_center_crop=False,
        )

        image_tools.extract_pano_faces(
            input_dir=config_values("original_sem_images_path", cfg),
            output_dir=config_values("semantic_images_path", cfg),
            input_file_type=config_values("sem_type", cfg),
            output_file_type=config_values("sem_output_type", cfg),
            run_number=config_values("run", cfg),
            filter=config_values("filter_semantic", cfg),
            apply_center_crop=False,
        )

    image_tools.find_all_trees(
        segmentid_dir=config_values("segmentid_images_path", cfg),
        color_dir=config_values("color_images_path", cfg),
        output_dir=config_values("output_trees_dir", cfg),
        tree_attributes_dict=tree_attributes_dict,
        semantic_dir=config_values("semantic_images_path", cfg),
        cover=config_values("mask", cfg),
        input_file_type="png"
    )
    output_trees_dir = config_values("output_trees_dir", cfg)

    image_tools.create_dataset(output_trees_dir, output_trees_dir, only_copy=False)

    #classes = []
    #for tree in os.listdir(output_trees_dir):
    #    filename = os.path.splitext(tree)[0]
    #    parts = filename.split("_")
    #    if len(parts) < 2:
    #        continue  # Skip files that don't match the expected pattern
    #    if parts[2] not in classes:
    #        classes.append(parts[2])
    #        os.makedirs(os.path.join(output_trees_dir, parts[2]), exist_ok=True)
    #    shutil.move(
    #        os.path.join(output_trees_dir, tree),
    #        os.path.join(os.path.join(output_trees_dir, parts[2]), tree),
    #    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
