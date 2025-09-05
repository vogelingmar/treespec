"""Create the Essen dataset from the RGB and SegmentID Images."""

import os
import shutil
import hydra
from hydra.core.config_store import ConfigStore

from treespec.utils import image_tools
from treespec.utils import matching_tools
from treespec.utils.matching_tools import create_dictionary

from treespec.conf.config_parser import new_create_essen_dataset_config_values as config_values
from treespec.conf.config import TreespecConfig

cs = ConfigStore.instance()
cs.store(name="treespec_config", node=TreespecConfig)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: TreespecConfig):
    """Script that creates the Essen dataset from RGB, SegmentID and SemanticClass images and inventory data."""
    date = os.path.basename(config_values("input_dir", cfg))

    for run in config_values("runs", cfg):

        if not config_values("pictures_extracted", cfg):
            if config_values("rgb_format", cfg) == "rectangle":
                image_tools.select_rgb_images(
                    input_dir=os.path.join(config_values("input_dir", cfg), "images"),
                    output_dir=os.path.join(config_values("output_dir", cfg), date, f"rgb_crops_{date}_{run}"),
                    image_file_type="png",
                )
            else:
                image_tools.extract_pano_faces(
                    input_dir=os.path.join(config_values("input_dir", cfg), "panos"),
                    output_dir=os.path.join(config_values("output_dir", cfg), date, f"rgb_crops_{date}_{run}"),
                    input_file_type="jpg",
                    output_file_type="png",
                    run_number=run,
                    apply_center_crop=config_values("rgb_format", cfg) == "rectangle",
                )

            image_tools.extract_pano_faces(
                    input_dir=os.path.join(config_values("input_dir", cfg), "panos", f"rend{run}"),
                    output_dir=os.path.join(config_values("output_dir", cfg), date, f"id_crops_{date}_{run}"),
                    input_file_type="tif",
                    output_file_type="png",
                    run_number=run,
                    filter="segmentid",
                    apply_center_crop=config_values("rgb_format", cfg) == "rectangle",
                )
            
            image_tools.extract_pano_faces(
                    input_dir=os.path.join(config_values("input_dir", cfg), "panos", f"rend{run}"),
                    output_dir=os.path.join(config_values("output_dir", cfg), date, f"sem_crops_{date}_{run}"),
                    input_file_type="tif",
                    output_file_type="png",
                    run_number=run,
                    filter="semanticclass",
                    apply_center_crop=config_values("rgb_format", cfg) == "rectangle",
                )
            
            matching_tools.match_and_export(predicted_inventory_path=os.path.join(config_values("input_dir", cfg), "inventory", f"run{run}", f"inventory{run}"),
                                            inventory_path=config_values("groundtruth_inventory_path", cfg),
                                            output_path=os.path.join(config_values("output_dir", cfg), date, f"inventory_{date}_{run}", f"inventory_{date}_{run}"),
                                            use_dbh_filter=False)

        tree_attributes_dict = create_dictionary(os.path.join(config_values("output_dir", cfg), date, f"inventory_{date}_{run}", f"inventory_{date}_{run}"))


        output_trees_dir = os.path.join(config_values("output_dir", cfg), f"trees_{run}")
        image_tools.find_all_trees(
            segmentid_dir=os.path.join(config_values("output_dir", cfg), f"id_crops_{run}"),
            color_dir=os.path.join(config_values("output_dir", cfg), f"rgb_crops_{run}"),
            output_dir=output_trees_dir,
            tree_attributes_dict=tree_attributes_dict,
            semantic_dir=os.path.join(config_values("output_dir", cfg), f"sem_crops_{run}"),
            cover=config_values("crop", cfg),
            input_file_type="png",
            run_number=run,
            date=date,
        )
        
        image_tools.create_dataset(output_trees_dir, output_trees_dir, only_copy=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
