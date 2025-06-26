"""Utility functions for processing images and combining images."""

import os
from typing import Optional
import shutil
import imageio.v2 as imageio
import numpy as np
import py360convert  # TODO: add to install requirements
from skimage.transform import resize  # TODO: add to install requirement


def select_rgb_images(input_dir: str, output_dir: str, image_type: str):
    """Selects and renames RGB images from an input directory and copies them to an output directory 
    based on their naming convention.

    Args:
        input_dir: Directory containing the input images.
        output_dir: Directory where the selected and renamed images will be saved.
        image_type: The file type of the images (e.g., 'jpg', 'png').
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(f"{1}.{image_type}") or file.endswith(f"{3}.{image_type}"):
            # Split filename to get the index before the last underscore
            name_wo_ext = os.path.splitext(file)[0]
            parts = name_wo_ext.split("_")
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


def extract_pano_faces(
    input_dir: str,
    output_dir: str,
    input_type: str,
    output_type: str,
    run: str,
    crop: bool,
    filter: Optional[str] = "",
):
    """Extracts left and right faces from panoramic images in the input directory 
    and saves them to the output directory.

    Args:
        input_dir: Directory containing the input panoramic images.
        output_dir: Directory where the extracted faces will be saved.
        input_type: The file type of the input images (e.g., 'jpg', 'png').
        output_type: The file type for the output images (e.g., 'jpg', 'png').
        run: A string to filter images based on their naming convention.
        crop: Whether to crop the faces to the center square.
        filter: Optional filter to select specific types of images (e.g., 'rgb', 'depth').
    """
    os.makedirs(output_dir, exist_ok=True)

    if filter is None or filter == "":
        type = "rgb"
    else:
        type = filter

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(f"{filter}.{input_type}"):
            name_wo_ext = os.path.splitext(file)[0]
            parts = name_wo_ext.split("_")
            if len(parts) < 2:
                continue  # Skip files that don't match the expected pattern
            if parts[1].endswith(str(run)):
                img = imageio.imread(os.path.join(input_dir, file))
                img = np.flip(img, axis=1)  # type: ignore #TODO: remove this line with new data
                cube_faces = py360convert.e2c(
                    img, face_w=4096, cube_format="list", mode="nearest"
                )  # returns list of 6 faces

                number = int(parts[2])
                for i, face in enumerate(cube_faces):
                    if i in (1, 3):
                        # Crop the face to the center square
                        height, width = face.shape[:2]
                        if crop:
                            start_y, end_y = height // 4, 3 * height // 4
                            start_x, end_x = width // 4, 3 * width // 4
                            cropped_face = face[start_y:end_y, start_x:end_x]
                        else:
                            cropped_face = face

                        if i == 1:
                            imageio.imwrite(
                                os.path.join(output_dir, f"{number}_{type}_left.{output_type}"),
                                cropped_face,
                            )
                        if i == 3:
                            imageio.imwrite(
                                os.path.join(
                                    output_dir,
                                    f"{number}_{type}_right.{output_type}",
                                ),
                                cropped_face,
                            )

    print(f"Extracted {type} faces from the panoramic images to {output_dir}")


def extract_tree_images(
    segmentid_face_path: str,
    color_face_path: str,
    output_dir: str,
    tree_attributes_dict: dict,
    cover: bool,
    image: str,
):
    """Extracts tree images from segmentid and color images based on segment IDs.
    Args:
        segmentid_face_path: Path to the segment ID face image.
        color_face_path: Path to the color face image.
        output_dir: Directory where the extracted tree images will be saved.
        tree_attributes_dict: Dictionary containing tree attributes.
        cover: Whether to apply a mask to the cropped images.
        image: Identifier for the image being processed (e.g., "image_number_orientation").
    """
    segmentid_face = imageio.imread(segmentid_face_path)
    color_face = imageio.imread(color_face_path)

    seg_h, seg_w = segmentid_face.shape[:2]
    col_h, col_w = color_face.shape[:2]

    unique_ids = np.unique(segmentid_face)
    for seg_id in unique_ids:
        if seg_id in (0, 1, 2):
            continue

        mask = segmentid_face == seg_id
        coords = np.argwhere(mask)
        if coords.size < 50 * 50:
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
            tree_species = tree_attributes_dict[float(seg_id)]["BAUMART"]
        else:
            tree_species = "unknown"

        # TODO:maybe change the id of the image to the BAUMID from the essen cadastre -> what happens to new trees?
        out_path = os.path.join(output_dir, f"{seg_id}_{image}_{tree_species}.png")

        if cover:
            # Resize mask to match cropped shape
            mask_cropped = mask[y0:y1, x0:x1]
            mask_resized = np.array(mask_cropped, dtype=np.uint8)
            if cropped.shape[:2] != mask_resized.shape:
                # If shapes don't match due to rounding, resize mask
                mask_resized = resize(
                    mask_cropped,
                    cropped.shape[:2],
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)

            # Apply mask: set everything outside mask to black
            if cropped.ndim == 3:
                masked_cropped = cropped * mask_resized[..., None]
            else:
                masked_cropped = cropped * mask_resized

            imageio.imwrite(out_path, masked_cropped)
        else:
            imageio.imwrite(out_path, cropped)


def find_all_trees(
    segmentid_dir: str,
    color_dir: str,
    output_dir: str,
    tree_attributes_dict: dict,
    cover: bool,
):
    """Extracts tree images from segmentid and color directories based on segment IDs.
    Args:
        segmentid_dir: Directory containing segment ID images.
        color_dir: Directory containing color images.
        output_dir: Directory where the extracted tree images will be saved.
        tree_attributes_dict: Dictionary containing tree attributes.
        cover: Whether to apply a mask to the cropped images.
    """
    os.makedirs(output_dir, exist_ok=True)
    for segmentid_image in os.listdir(segmentid_dir):
        name_wo_ext = os.path.splitext(segmentid_image)[0]
        parts = name_wo_ext.split("_")
        if len(parts) < 2:
            continue  # Skip files that don't match the expected pattern
        image_number = parts[0]
        orientation = parts[2]
        color_path = os.path.join(color_dir, f"{image_number}_rgb_{orientation}.jpg")
        segmentid_path = os.path.join(segmentid_dir, segmentid_image)
        extract_tree_images(
            segmentid_face_path=segmentid_path,
            color_face_path=color_path,
            output_dir=output_dir,
            tree_attributes_dict=tree_attributes_dict,
            cover=cover,
            image=f"{image_number}{orientation}",
        )

    print(f"Extracted tree images to {output_dir}")
