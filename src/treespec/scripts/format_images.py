import os
import shutil
import imageio.v2 as imageio
import numpy as np
import py360convert

from treespec.scripts.matching import create_dictionary


def select_rgb_images(input_dir: str, output_dir: str, image_type: str):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(f"{1}.{image_type}") or file.endswith(f"{3}.{image_type}"):
            # Split filename to get the index before the last underscore
            name_wo_ext = os.path.splitext(file)[0]
            parts = name_wo_ext.split('_')
            if len(parts) < 2:
                continue  # Skip files that don't match the expected pattern
            idx = parts[-2]
            if file.endswith(f"{1}.{image_type}"):
                new_name = f"{idx}_rgb_left.{image_type}"
            else:
                new_name = f"{idx}_rgb_right.{image_type}"
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, new_name)
            shutil.copy2(input_path, output_path)
    
    print(f"Copied images to {output_dir}")

def extract_segmentid_faces(input_dir: str, output_dir: str, input_type: str, output_type: str, run: str):
    os.makedirs(output_dir, exist_ok=True)

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(f"segmentid.{input_type}"):
            name_wo_ext = os.path.splitext(file)[0]
            parts = name_wo_ext.split('_')
            if len(parts) < 2:
                continue  # Skip files that don't match the expected pattern
            if parts[1].endswith(run):
                img = imageio.imread(os.path.join(input_dir, file))
                img = np.flip(img, axis=1)
                cube_faces = py360convert.e2c(img, face_w=500, cube_format='list', mode='nearest')  # returns list of 6 faces

                number = int(parts[2])
                for i, face in enumerate(cube_faces):
                    if i == 1 or i == 3:
                        # Crop the face to the center square
                        height, width = face.shape[:2]
                        start_y, end_y = height // 4, 3 * height // 4
                        start_x, end_x = width // 4, 3 * width // 4
                        cropped_face = face[start_y:end_y, start_x:end_x]

                        if i == 1: 
                            imageio.imwrite(os.path.join(output_dir, f"{number}_segmentid_left.{output_type}"), cropped_face)
                        if i == 3:
                            imageio.imwrite(os.path.join(output_dir, f"{number}_segmentid_right.{output_type}"), cropped_face)

    print(f"Extracted segment ID faces from the panoramic images to {output_dir}")

def extract_tree_images(segmentid_face_path: str, color_face_path: str, output_dir: str, tree_attributes_dict: dict):
    
    segmentid_face = imageio.imread(segmentid_face_path)
    color_face = imageio.imread(color_face_path)

    seg_h, seg_w = segmentid_face.shape[:2]
    col_h, col_w = color_face.shape[:2]


    unique_ids = np.unique(segmentid_face)
    for seg_id in unique_ids:
        if seg_id == 0 or seg_id == 1 or seg_id == 2:
            continue

        mask = segmentid_face == seg_id
        coords = np.argwhere(mask)
        if coords.size < 50*50:
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # +1 for slicing

        # Calculate relative coordinates
        rel_y0, rel_x0 = y0 / seg_h, x0 / seg_w
        rel_y1, rel_x1 = y1 / seg_h, x1 / seg_w

        # Map to color_face coordinates
        col_y0 = int(rel_y0 * col_h)
        col_x0 = int(rel_x0 * col_w)
        col_y1 = int(rel_y1 * col_h)
        col_x1 = int(rel_x1 * col_w)

        # Crop from color image
        cropped = color_face[col_y0:col_y1, col_x0:col_x1]

        if float(seg_id) in tree_attributes_dict.keys():
            tree_species = tree_attributes_dict[float(seg_id)]['BAUMART']
        else:
            tree_species = "unknown"

        # Save the cropped image
        out_path = os.path.join(output_dir, f"{seg_id}_tree_{tree_species}.png")
        imageio.imwrite(out_path, cropped)

def extract_trees(segmentid_dir, color_dir, output_dir, tree_attributes_dict):
    os.makedirs(output_dir, exist_ok=True)
    for segmentid_image in os.listdir(segmentid_dir):
        name_wo_ext = os.path.splitext(segmentid_image)[0]
        parts = name_wo_ext.split('_')
        if len(parts) < 2:
            continue  # Skip files that don't match the expected pattern
        color_path = os.path.join(color_dir, f"{parts[0]}_rgb_{parts[2]}.jpg")
        segmentid_path = os.path.join(segmentid_dir, segmentid_image)
        extract_tree_images(segmentid_face_path=segmentid_path,
                            color_face_path=color_path,
                            output_dir=output_dir,
                            tree_attributes_dict=tree_attributes_dict)
    print(f"Extracted tree images to {output_dir}")




basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
attribute_path = "/data/essen/cadastre/matched_output/matched_output"

tree_attributes_dict = create_dictionary(attribute_path)

select_rgb_images("/data/essen/dataset/color/Run 42 Camera 4 360", 
              os.path.join(basepath, "io/pictures/color_42"), 
              "jpg")

extract_segmentid_faces("/data/essen/dataset/depth and seg/run_42", 
                      os.path.join(basepath, "io/pictures/segmentid_42"), 
                      "tif",
                      "png",
                      "42")

extract_trees(os.path.join(basepath, "io/pictures/segmentid_42"),
              os.path.join(basepath, "io/pictures/color_42"),
              os.path.join(basepath, "io/pictures/trees_42"),
              tree_attributes_dict)
