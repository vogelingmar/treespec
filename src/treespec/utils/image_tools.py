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
    input_pano_file_type: str,
    output_image_file_type: str,
    run_number: int,
    apply_center_crop: bool,
    name_filter: Optional[str] = "",
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

    if name_filter is None or name_filter == "":
        image_type = "rgb"
    else:
        image_type = name_filter

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(f"{name_filter}.{input_pano_file_type}"):
            filename = os.path.splitext(file)[0]
            parts = filename.split("_")
            if len(parts) < 2:
                continue
            pano_run_number = parts[1]
            if pano_run_number.endswith(str(run_number)):
                pano = imageio.imread(os.path.join(input_dir, file))
                pano_width = pano.shape[1]
                # Set face_w to one fourth of the panorama width, but at least 500px
                face_width = max(pano_width // 4, 500)

                cube_faces = py360convert.e2c(
                    pano, face_w=face_width, cube_format="list", mode="nearest"
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
                            output_dir, f"{filename_prefix}_{face_label}.{output_image_file_type}"
                        )
                        imageio.imwrite(output_path, face)

    print(f"Extracted {image_type} left and right faces from the panoramic images to {output_dir}")

def determine_output_path(
    tree_id, tree_attributes_dict, output_dir, image_id, cover
) -> str:
    r"""
    Determines the output path for a tree image based on its ID and species."""
    if str(tree_id) in tree_attributes_dict.keys():
        tree_species = tree_attributes_dict[str(tree_id)]["BAUMART"]
    else:
        tree_species = "unknown"
    return os.path.join(output_dir, cover, f"{tree_id}_{image_id}_{tree_species}.png")

def extract_relevant_patch(mask, color_face):
    r"""Extracts the relevant zoomed and cropped patches from the color face based on the provided mask."""

    crop_coords = np.argwhere(mask)
    if crop_coords.size < 200 * 200 or crop_coords.shape[0] < crop_coords.shape[1]: # filter small and wide images
        return
    
    crop_bound_y0, crop_bound_x0 = crop_coords.min(axis=0)
    crop_bound_y1, crop_bound_x1 = crop_coords.max(axis=0) + 1  # +1 for slicing

    zoomed_color_face = color_face[crop_bound_y0:crop_bound_y1, crop_bound_x0:crop_bound_x1]
    zoomed_mask = mask[crop_bound_y0:crop_bound_y1, crop_bound_x0:crop_bound_x1] # gives cropped mask of whole tree

    zoomed_cropped_color_face = zoomed_color_face * zoomed_mask[..., None]

    return zoomed_cropped_color_face, zoomed_color_face, zoomed_mask, crop_bound_y0, crop_bound_y1, crop_bound_x0, crop_bound_x1

def determine_colored_area(zoomed_cropped_color_face):
    r"""Determines the proportion of colored (non-black) area in the zoomed and cropped color face for filtering."""
    colored = np.count_nonzero(np.any(zoomed_cropped_color_face != 0, axis=-1))
    total = zoomed_cropped_color_face.shape[0] * zoomed_cropped_color_face.shape[1]
    return colored / total

def create_tree_image(tree_id, tree_attributes_dict, segmentid_face, color_face, semanticclass_face, output_dir, cover, image_id):
    r"""Creates, filters and saves a tree or bark image based on the provided parameters and masks."""

    id_mask = segmentid_face == tree_id # binary mask for the current tree id
    patch_result = extract_relevant_patch(id_mask, color_face)
    if patch_result is None:
        return  # Skip this tree, nothing to extract

    zoomed_cropped_tree_face, zoomed_tree_face, zoomed_id_mask, id_bound_y0, id_bound_y1, id_bound_x0, id_bound_x1 = patch_result
    
    semanticclass_zoomed = semanticclass_face[id_bound_y0:id_bound_y1, id_bound_x0:id_bound_x1]

    zoomed_bark_mask = (semanticclass_zoomed == 1).astype(np.uint8) & zoomed_id_mask # binary mask for bark only within the current tree id
    patch_result = extract_relevant_patch(zoomed_bark_mask, zoomed_tree_face)
    if patch_result is None:
        return
    
    zoomed_cropped_bark_face, zoomed_bark_face, zoomed_bark_mask, _, _, _, _ = patch_result

    if determine_colored_area(zoomed_cropped_bark_face) < 0.5:
        return
    
    covers = ["tree", "tree_crop", "bark", "bark_crop"]
    for coverer in covers:
        output_image = None
        match coverer:
            case "tree":
                output_image = zoomed_tree_face
            case "tree_crop":
                output_image = zoomed_cropped_tree_face
            case "bark":
                output_image = zoomed_bark_face
            case "bark_crop":
                output_image = zoomed_cropped_bark_face
        os.makedirs(os.path.join(output_dir, coverer), exist_ok=True)
        imageio.imwrite(determine_output_path(tree_id, tree_attributes_dict, output_dir, image_id, coverer), output_image)

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
    semanticclass_face = imageio.imread(semanticclass_face_path)

    # Resize segmentid_face to match color_face resolution using nearest-neighbor interpolation
    if segmentid_face.shape[:2] != color_face.shape[:2]:
        segmentid_face = resize(
            segmentid_face,
            color_face.shape[:2],
            order=0,  # nearest-neighbor
            preserve_range=True,
            anti_aliasing=False
        ).astype(segmentid_face.dtype)
    if semanticclass_face.shape[:2] != color_face.shape[:2]:
        semanticclass_face = resize(
            semanticclass_face,
            color_face.shape[:2],
            order=0,  # nearest-neighbor
            preserve_range=True,
            anti_aliasing=False
        ).astype(semanticclass_face.dtype)

    tree_ids = np.unique(segmentid_face)
    for tree_id in tree_ids:
        if tree_id not in (0, 1): # skip background and ground
            create_tree_image(
                tree_id=tree_id,
                tree_attributes_dict=tree_attributes_dict,
                color_face=color_face,
                segmentid_face=segmentid_face,
                semanticclass_face=semanticclass_face,
                output_dir=output_dir,
                cover=cover,
                image_id=image_id,
            )


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