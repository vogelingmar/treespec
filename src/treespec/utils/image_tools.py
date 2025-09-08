"""Utility functions for processing images and combining images."""

import os
from typing import Optional
import shutil
import imageio.v2 as imageio
import numpy as np
import py360convert
from skimage.transform import resize


def select_rgb_images(input_dir: str, output_dir: str, image_file_type: str):
    r"""Selects and renames RGB images from an input directory and copies them to an output directory
    based on their naming convention.

    Args:
        input_dir: Directory containing the input images.
        output_dir: Directory where the selected and renamed images will be saved.
        image_file_type: The file type of the images (e.g., 'jpg', 'png').
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(f"{1}.{image_file_type}") or file.endswith(f"{3}.{image_file_type}"):
            name_wo_ext = os.path.splitext(file)[0]
            parts = name_wo_ext.split("_")
            if len(parts) < 2:
                continue
            idx = parts[-2]
            if file.endswith(f"{1}.{image_file_type}"):
                new_name = f"{idx}_rgb_left.{image_file_type}"
            else:
                new_name = f"{idx}_rgb_right.{image_file_type}"
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, new_name)
            shutil.copy2(input_path, output_path)

    print(f"Copied images to {output_dir}")

def extract_pano_faces(
    input_dir: str,
    output_dir: str,
    input_file_type: str,
    output_file_type: str,
    run_number: int,
    apply_center_crop: bool,
    filter: Optional[str] = "",
):
    r"""Extracts left and right faces from panoramic images in the input directory
    and saves them to the output directory.

    Args:
        input_dir: Directory containing the input panoramic images.
        output_dir: Directory where the extracted faces will be saved.
        input_file_type: The file type of the input images (e.g., 'jpg', 'png').
        output_file_type: The file type for the output images (e.g., 'jpg', 'png').
        run_number: The number of the recording run to filter images accordingly.
        apply_center_crop: Whether to crop the faces to the center square (apply when using square rgb images).
        filter: Optional filter to select specific types of images (e.g. 'segmentid', 'semanticclass'). If left empty, type = 'rgb' is assumed.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filter is None or filter == "":
        image_type = "rgb"
    else:
        image_type = filter

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(f"{filter}.{input_file_type}"):
            filename = os.path.splitext(file)[0]
            parts = filename.split("_")
            if len(parts) < 2:
                continue
            if parts[1].endswith(str(run_number)):
                img = imageio.imread(os.path.join(input_dir, file))
                img_h, img_w = img.shape[:2]
                # Set face_w to one fourth of the panorama width, but at least 500px
                face_w = max(img_w // 4, 500)

                cube_faces = py360convert.e2c(
                    img, face_w=face_w, cube_format="list", mode="nearest"
                )

                image_number = int(parts[2])
                for i, face in enumerate(cube_faces):
                    if i in (1, 3):  # 1 = left, 3 = right
                        height, width = face.shape[:2]
                        if apply_center_crop:
                            start_y, end_y = height // 4, 3 * height // 4
                            start_x, end_x = width // 4, 3 * width // 4
                            face = face[start_y:end_y, start_x:end_x]

                        # If semanticclass, extract red channel only
                        if image_type == "semanticclass":
                            # Ensure face has at least 3 channels
                            if face.ndim == 3 and face.shape[2] >= 1:
                                face = face[:, :, 0]  # R channel

                        filename_prefix = f"{image_number}_{image_type}"
                        face_label = "left" if i == 1 else "right"
                        output_path = os.path.join(
                            output_dir, f"{filename_prefix}_{face_label}.{output_file_type}"
                        )
                        imageio.imwrite(output_path, face)

    print(f"Extracted {image_type} left and right faces from the panoramic images to {output_dir}")


def extract_tree_images(
    color_face_path: str,
    segmentid_face_path: str,
    semanticclass_face_path: Optional[str],
    output_dir: str,
    tree_attributes_dict: dict,
    cover: Optional[str],
    image_id: str,
):
    r"""Extracts tree/bark images from segmentid, color and semanticclass images and optionally masks out certain areas.

    Args:
        color_face_path: Path to the color face image.
        segmentid_face_path: Path to the segment ID face image.
        semantic_face_path: Path to the semantic image.
        output_dir: Directory where the extracted tree images will be saved.
        tree_attributes_dict: Dictionary containing tree attributes.
        cover: Extraction mode: None, 'tree', 'tree_crop', 'bark', or 'bark_crop'.
        image_id: Identifier for the image being processed.
    """
    color_face = imageio.imread(color_face_path)
    segmentid_face = imageio.imread(segmentid_face_path)

    # Resize segmentid_face to match color_face resolution using nearest-neighbor interpolation
    if segmentid_face.shape[:2] != color_face.shape[:2]:
        segmentid_face = resize(
            segmentid_face,
            color_face.shape[:2],
            order=0,  # nearest-neighbor
            preserve_range=True,
            anti_aliasing=False
        ).astype(segmentid_face.dtype)

    tree_ids = np.unique(segmentid_face)
    for tree_id in tree_ids:
        if tree_id in (0, 1): # skip background and ground
            continue
        elif str(tree_id) in tree_attributes_dict.keys():
            tree_species = tree_attributes_dict[str(tree_id)]["BAUMART"]
        else:
            tree_species = "unknown"
        out_path = os.path.join(output_dir, f"{tree_id}_{image_id}_{tree_species}.png")

        id_mask = segmentid_face == tree_id # binary mask for the current tree id
        id_coords = np.argwhere(id_mask)
        if id_coords.size < 200 * 200 or id_coords.shape[0] < id_coords.shape[1]: # filter very small trees
            continue
        id_bound_y0, id_bound_x0 = id_coords.min(axis=0)
        id_bound_y1, id_bound_x1 = id_coords.max(axis=0) + 1  # +1 for slicing
        zoomed_tree_face = color_face[id_bound_y0:id_bound_y1, id_bound_x0:id_bound_x1]

        id_mask_zoomed = id_mask[id_bound_y0:id_bound_y1, id_bound_x0:id_bound_x1] # gives cropped mask of whole tree

        #TODO: add filtering here to only include trees where both tree crown and trunk are visible
        output_image = zoomed_tree_face

        if zoomed_tree_face.ndim == 3:
            zoomed_cropped_tree_face = zoomed_tree_face * id_mask_zoomed[..., None]
        else:
            zoomed_cropped_tree_face = zoomed_tree_face * id_mask_zoomed

        if cover == "tree_crop":
            output_image = zoomed_cropped_tree_face

        elif cover in ["bark", "bark_crop"]:
            if semanticclass_face_path is None:
                raise ValueError("To extract only the barks from the image a semantic face is required!")
            else:
                semanticclass_face = imageio.imread(semanticclass_face_path)
                if semanticclass_face.shape[:2] != color_face.shape[:2]:
                    semanticclass_face = resize(
                        semanticclass_face,
                        color_face.shape[:2],
                        order=0,  # nearest-neighbor
                        preserve_range=True,
                        anti_aliasing=False
                    ).astype(semanticclass_face.dtype)
                semanticclass_zoomed = semanticclass_face[id_bound_y0:id_bound_y1, id_bound_x0:id_bound_x1]

                bark_mask = (semanticclass_zoomed == 1).astype(np.uint8) & id_mask_zoomed # binary mask for bark only within the current tree id
                bark_coords = np.argwhere(bark_mask)
                if bark_coords.size < 200 * 200 or bark_coords.shape[0] < bark_coords.shape[1]: # filter very small trees
                    continue
                bark_bound_y0, bark_bound_x0 = bark_coords.min(axis=0)
                bark_bound_y1, bark_bound_x1 = bark_coords.max(axis=0) + 1  # +1 for slicing
                zoomed_bark_face = zoomed_tree_face[bark_bound_y0:bark_bound_y1, bark_bound_x0:bark_bound_x1]

                output_image = zoomed_bark_face

                zoomed_bark_mask = bark_mask[bark_bound_y0:bark_bound_y1, bark_bound_x0:bark_bound_x1]
                zoomed_tree_cropped_bark_face = zoomed_cropped_tree_face[bark_bound_y0:bark_bound_y1, bark_bound_x0:bark_bound_x1]
                if zoomed_tree_cropped_bark_face.ndim == 3:
                    zoomed_cropped_bark_face = zoomed_tree_cropped_bark_face * zoomed_bark_mask[..., None]
                else:
                    zoomed_cropped_bark_face = zoomed_tree_cropped_bark_face * zoomed_bark_mask
                if zoomed_cropped_bark_face.ndim == 3:
                    non_black = np.count_nonzero(np.any(zoomed_cropped_bark_face != 0, axis=-1))
                    total = zoomed_cropped_bark_face.shape[0] * zoomed_cropped_bark_face.shape[1]
                else:
                    non_black = np.count_nonzero(zoomed_cropped_bark_face != 0)
                    total = zoomed_cropped_bark_face.size
                if non_black / total < 0.5: # filter images with too much black area
                    continue

                if cover == "bark_crop":
                    output_image = zoomed_cropped_bark_face

        imageio.imwrite(out_path, output_image)


def find_all_trees(
    segmentid_dir: str,
    color_dir: str,
    output_dir: str,
    tree_attributes_dict: dict,
    run_number: int,
    date: str,
    input_file_type: str = "png",
    cover: Optional[str] = None,
    semantic_dir: Optional[str] = None,
):
    r"""Finds and extracts tree images from segment ID, color, and semantic images from the specified directories by matching their IDs.

    Args:
        segmentid_dir: Directory containing segment ID images.
        color_dir: Directory containing color images.
        output_dir: Directory where the extracted tree images will be saved.
        tree_attributes_dict: Dictionary containing tree attributes.
        input_file_type: The file type of the input images (e.g., 'png', 'jpg').
        cover: Whether to apply a mask to the cropped images.
        semantic_dir: Directory containing semantic images.

    Raises:
        ValueError: If `cover` is "bark" and `semantic_dir` is None.
    """
    if semantic_dir == None and cover == "bark":
        raise ValueError(
            "To extract only the barks from the images, semantic images are required! Give a semantic dir."
        )

    os.makedirs(output_dir, exist_ok=True)
    for segmentid_image in os.listdir(segmentid_dir):
        filename = os.path.splitext(segmentid_image)[0]
        parts = filename.split("_")
        if len(parts) < 2:
            continue
        image_number = parts[0]
        orientation = parts[2]
        color_path = os.path.join(color_dir, f"{image_number}_rgb_{orientation}.{input_file_type}")
        semantic_path = (
            os.path.join(
                semantic_dir,
                f"{image_number}_semanticclass_{orientation}.{input_file_type}",
            )
            if semantic_dir
            else None
        )
        segmentid_path = os.path.join(segmentid_dir, segmentid_image)
        extract_tree_images(
            segmentid_face_path=segmentid_path,
            color_face_path=color_path,
            output_dir=output_dir,
            tree_attributes_dict=tree_attributes_dict,
            cover=cover,
            semanticclass_face_path=semantic_path,
            image_id=f"{date}_{run_number}_{image_number}_{orientation}",
        )

    print(f"Extracted tree images to {output_dir}")


def create_dataset(input_trees_dir: str, output_dataset_dir: str, only_copy: bool):
    r"""Creates a dataset from the extracted tree images based on their names.

    Args:
        input_trees_dir: Directory where the pictures for the dataset are stored.
        output_dataset_dir: Directory where the dataset will be created.
        only_copy: If True, copies the files; if False, moves them.
    """
    classes = []
    for tree in os.listdir(input_trees_dir):
        filename = os.path.splitext(tree)[0]
        parts = filename.split("_")
        if len(parts) < 5:
            continue
        species = parts[5]
        if species not in classes:
            classes.append(species)
            os.makedirs(os.path.join(output_dataset_dir, species), exist_ok=True)
        if only_copy:
            shutil.copy2(
                os.path.join(input_trees_dir, tree),
                os.path.join(os.path.join(output_dataset_dir, species), tree),
            )
        else:
            shutil.move(
                os.path.join(input_trees_dir, tree),
                os.path.join(os.path.join(output_dataset_dir, species), tree),
            )
    print(f"Created dataset with {len(classes)} classes in {output_dataset_dir}")