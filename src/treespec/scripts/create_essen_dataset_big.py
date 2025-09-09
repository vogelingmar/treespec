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


@hydra.main(config_path="../conf", config_name="big_dataset_creation_config")
def main(cfg: TreespecConfig):
    """Script that creates the Essen dataset from RGB, SegmentID and SemanticClass images and inventory data."""
    basedir = os.listdir(cfg.create_big_essen_dataset.input_dir)

    for date_dir in basedir:
        if date_dir.startswith("2022-09"):
            date = date_dir
            if date in cfg.create_big_essen_dataset.date_and_runs.keys():
                for run in cfg.create_big_essen_dataset.date_and_runs[date]:
                    image_tools.extract_pano_faces(
                        input_dir=os.path.join(cfg.create_big_essen_dataset.input_dir, date, "panos"),
                        output_dir=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"rgb_crops_{date}_{run}"),
                        input_pano_file_type="jpg",
                        output_image_file_type="png",
                        run_number=run,
                        apply_center_crop=False,
                    )
                    image_tools.extract_pano_faces(
                            input_dir=os.path.join(cfg.create_big_essen_dataset.input_dir, date, "panos", f"rend{run}"),
                            output_dir=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"id_crops_{date}_{run}"),
                            input_pano_file_type="tif",
                            output_image_file_type="png",
                            run_number=run,
                            name_filter="segmentid",
                            apply_center_crop=False,
                        )

                    image_tools.extract_pano_faces(
                            input_dir=os.path.join(cfg.create_big_essen_dataset.input_dir, date, "panos", f"rend{run}"),
                            output_dir=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"sem_crops_{date}_{run}"),
                            input_pano_file_type="tif",
                            output_image_file_type="png",
                            run_number=run,
                            name_filter="semanticclass",
                            apply_center_crop=False,
                        )
            
                    matching_tools.match_and_export(predicted_inventory_path=os.path.join(cfg.create_big_essen_dataset.input_dir, date, "inventory", f"run{run}", f"inventory{run}"),
                                                    inventory_path=cfg.create_big_essen_dataset.groundtruth_inventory_path,
                                                    output_path=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"inventory_{date}_{run}", f"inventory_{date}_{run}"),
                                                    use_dbh_filter=False)

                    tree_attributes_dict = create_dictionary(os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"inventory_{date}_{run}", f"inventory_{date}_{run}"))


                    output_trees_dir = os.path.join(cfg.create_big_essen_dataset.output_dir, "datasets")
                    image_tools.find_all_trees(
                        segmentid_dir=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"id_crops_{date}_{run}"),
                        color_dir=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"rgb_crops_{date}_{run}"),
                        output_dir=output_trees_dir,
                        tree_attributes_dict=tree_attributes_dict,
                        semantic_dir=os.path.join(cfg.create_big_essen_dataset.output_dir, "preprocessing", date, f"sem_crops_{date}_{run}"),
                        cover="tree",
                        input_file_type="png",
                        run_number=run,
                        date=date,
                    )
    
    for dataset in os.listdir(output_trees_dir):
        directory = os.path.join(output_trees_dir, dataset)
        image_tools.create_dataset(directory, directory, only_copy=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
