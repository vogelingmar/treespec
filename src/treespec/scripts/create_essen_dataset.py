import os
import shutil

import treespec.scripts.image_tools as image_tools
from treespec.scripts.matching import create_dictionary

from treespec.conf.config_parser import create_essen_dataset_config_values as config_values

if __name__ == "__main__":

    tree_attributes_dict = create_dictionary(config_values["attribute_path"])

    image_tools.select_rgb_images(input_dir=config_values["original_color_images_path"], 
                                output_dir=config_values["color_images_path"], 
                                image_type=config_values["color_type"])

    image_tools.extract_pano_faces(input_dir=config_values["original_seg_images_path"], 
                                        output_dir=config_values["segmentid_images_path"], 
                                        input_type=config_values["seg_type"], 
                                        output_type=config_values["seg_output_type"], 
                                        run=config_values["run"],
                                        filter=config_values["filter"])

    image_tools.extract_trees(segmentid_dir=config_values["segmentid_images_path"],
                color_dir=config_values["color_images_path"],
                output_dir=config_values["output_trees_dir"],
                tree_attributes_dict=tree_attributes_dict,
                cover=config_values["mask"])

    output_trees_dir = config_values["output_trees_dir"]

    classes = []
    for tree in os.listdir(output_trees_dir):
        name_wo_ext = os.path.splitext(tree)[0]
        parts = name_wo_ext.split('_')
        if len(parts) < 2:
            continue  # Skip files that don't match the expected pattern
        if parts[2] not in classes:
            classes.append(parts[2])
            os.makedirs(os.path.join(output_trees_dir, parts[2]), exist_ok=True)
        shutil.move(os.path.join(output_trees_dir, tree), os.path.join(os.path.join(output_trees_dir, parts[2]), tree))


