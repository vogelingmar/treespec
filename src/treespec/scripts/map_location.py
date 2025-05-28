import numpy as np
import os

import py360convert
import imageio.v2 as imageio

from treespec.scripts.matching import create_dictionary

basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
depth_path = os.path.join(basepath, "io/helpers/depth_image.png")
segmentid_path = os.path.join(basepath, "io/helpers/segmentid_image.png")
color_path = os.path.join(basepath, "io/helpers/color_image.jpg")
attribute_path = "/data/essen/cadastre/matched_output"

tree_attributes_dict = create_dictionary(attribute_path)

output_dir = os.path.join(basepath, "io/pictures")
os.makedirs(os.path.dirname(output_dir), exist_ok=True)
output_bounding_box_dir = os.path.join(output_dir, "bounding_boxes_face1")
os.makedirs(output_bounding_box_dir, exist_ok=True)

def get_cube_faces(image_path, type):

    img = imageio.imread(image_path)
    img = np.flip(img, axis=1)
    cube_faces = py360convert.e2c(img, face_w=500, cube_format='list', mode='nearest')  # returns list of 6 faces

    for i, face in enumerate(cube_faces):
        if i == 1 or i == 3:
            height, width = face.shape[:2]
            start_y, end_y = height // 4, 3 * height // 4
            start_x, end_x = width // 4, 3 * width // 4

            cropped_face = face[start_y:end_y, start_x:end_x]

            #upsampled_face = cv2.resize(cropped_face, (2048, 2048), interpolation=cv2.INTER_CUBIC)

            imageio.imwrite(os.path.join(output_dir, f"{type}_cube_face_{i}.png"), cropped_face)


def extract_and_save_bounding_boxes(segmentid_face_path, color_face_path, output_dir):
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
        out_path = os.path.join(output_dir, f"tree_{seg_id}_bbox_({tree_species}).png")
        imageio.imwrite(out_path, cropped)
    
get_cube_faces(depth_path, "depth")
get_cube_faces(segmentid_path, "segmentid")

extract_and_save_bounding_boxes(os.path.join(basepath, "io/pictures/segmentid_cube_face_1.png"), color_path, output_bounding_box_dir)


# map the prediction to the tree centre of pointcloud
# pointtree extracts position of each tree id: map the prediction to the tree id in the cadastre
# for dataset creation match cadastre and pointcloud and take tree species from cadastre and put according tree id in folder

# pointcloud cadastre mapping (also import for the entire bachelors project): based on location and tree diameter or similar