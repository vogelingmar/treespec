import os
import shutil

import treespec.scripts.image_tools as image_tools
from treespec.scripts.matching import create_dictionary


basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
attribute_path = "/data/essen/cadastre/matched_output/matched_output"

original_color_images_path = "/data/essen/dataset/color/Run 40 Camera 4 360"
color_images_path = os.path.join(basepath, "io/pictures/color_40")
color_type = "jpg"

original_seg_images_path = "/data/essen/dataset/depth and seg/run_40"
segmentid_images_path = os.path.join(basepath, "io/pictures/segmentid_40")
seg_type = "tif"
seg_output_type = "png"
run = "40"

output_trees_dir = os.path.join(basepath, "io/pictures/trees_40")
tree_attributes_dict = create_dictionary(attribute_path)
mask = False


image_tools.select_rgb_images(input_dir=original_color_images_path, output_dir=color_images_path, image_type=color_type)

image_tools.extract_segmentid_faces(input_dir=original_seg_images_path, output_dir=segmentid_images_path, input_type=seg_type, output_type=seg_output_type, run=run)

image_tools.extract_trees(segmentid_dir=segmentid_images_path,
              color_dir=color_images_path,
              output_dir=output_trees_dir,
              tree_attributes_dict=tree_attributes_dict,
              cover=mask)

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


