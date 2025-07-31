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
                cube_faces = py360convert.e2c(
                    img, face_w=4096, cube_format="list", mode="nearest"
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
    semantic_face_path: Optional[str],
    output_dir: str,
    tree_attributes_dict: dict,
    cover: Optional[str],
    image_number: str,
):
    r"""Extracts tree/ bark images from segmentid, color and semanticclass images and optionaly masks out certain areas.

    Args:
        color_face_path: Path to the color face image.
        segmentid_face_path: Path to the segment ID face image.
        semantic_face_path: Path to the semantic image.
        output_dir: Directory where the extracted tree images will be saved.
        tree_attributes_dict: Dictionary containing tree attributes.
        cover: Whether to mask out certain areas in the images.
        image_number: Identifier for the image being processed.
    """
    color_face = imageio.imread(color_face_path)
    segmentid_face = imageio.imread(segmentid_face_path)

    if semantic_face_path == None and cover == "bark":
        raise ValueError("To extract only the barks from the image a semantic face is required!")

    seg_h, seg_w = segmentid_face.shape[:2]
    col_h, col_w = color_face.shape[:2]

    unique_ids = np.unique(segmentid_face)
    for seg_id in unique_ids:
        if seg_id in (0, 1):
            continue

        mask = segmentid_face == seg_id
        coords = np.argwhere(mask)
        if coords.size < 100 * 100:
            continue

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # +1 for slicing

        rel_y0, rel_x0 = y0 / seg_h, x0 / seg_w
        rel_y1, rel_x1 = y1 / seg_h, x1 / seg_w

        col_y0 = int(rel_y0 * col_h)
        col_x0 = int(rel_x0 * col_w)
        col_y1 = int(rel_y1 * col_h)
        col_x1 = int(rel_x1 * col_w)

        cropped = color_face[col_y0:col_y1, col_x0:col_x1]

        if float(seg_id) in tree_attributes_dict.keys():
            tree_species = tree_attributes_dict[float(seg_id)]["BAUMART"]
        else:
            tree_species = "unknown"

        out_path = os.path.join(output_dir, f"{seg_id}_{image_number}_{tree_species}.png")

        if cover in ["tree", "bark"]:
            mask_cropped = mask[y0:y1, x0:x1]
            mask_resized = np.array(mask_cropped, dtype=np.uint8)
            if cropped.shape[:2] != mask_resized.shape:
                mask_resized = resize(
                    mask_cropped,
                    cropped.shape[:2],
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)

            if cropped.ndim == 3:
                masked_cropped = cropped * mask_resized[..., None]
            else:
                masked_cropped = cropped * mask_resized

            if cover == "bark":
                semantic_face = imageio.imread(semantic_face_path)  # type: ignore
                sem_h, sem_w = semantic_face.shape[:2]
                sem_y0 = int(rel_y0 * sem_h)
                sem_x0 = int(rel_x0 * sem_w)
                sem_y1 = int(rel_y1 * sem_h)
                sem_x1 = int(rel_x1 * sem_w)

                sem_crop = semantic_face[sem_y0:sem_y1, sem_x0:sem_x1]
                if sem_crop.shape[:2] != masked_cropped.shape[:2]:
                    sem_crop = resize(
                        sem_crop,
                        masked_cropped.shape[:2],
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(semantic_face.dtype)
                bark_mask = (sem_crop == 1).astype(np.uint8)
                if masked_cropped.ndim == 3:
                    masked_cropped = masked_cropped * bark_mask[..., None]
                else:
                    masked_cropped = masked_cropped * bark_mask

                bark_coords = np.argwhere(
                    (bark_mask > 0)
                    & (np.any(masked_cropped != 0, axis=-1) if masked_cropped.ndim == 3 else masked_cropped != 0)
                )
                if bark_coords.size > 0:
                    bark_y0, bark_x0 = bark_coords.min(axis=0)
                    bark_y1, bark_x1 = bark_coords.max(axis=0) + 1  # +1 for slicing
                    if masked_cropped.ndim == 3:
                        masked_cropped = masked_cropped[bark_y0:bark_y1, bark_x0:bark_x1, :]
                        non_black = np.count_nonzero(np.any(masked_cropped != 0, axis=-1))
                        total = masked_cropped.shape[0] * masked_cropped.shape[1]
                    else:
                        masked_cropped = masked_cropped[bark_y0:bark_y1, bark_x0:bark_x1]
                        non_black = np.count_nonzero(masked_cropped != 0)
                        total = masked_cropped.size

                    # filter results to guarantee quality
                    min_size = 200
                    if (
                        masked_cropped.shape[0] < min_size
                        or masked_cropped.shape[1] < min_size
                        or masked_cropped.shape[0] < masked_cropped.shape[1]
                        or non_black / total < 0.5
                    ):
                        continue
                else:
                    continue

            imageio.imwrite(out_path, masked_cropped)
        elif cover is None:
            imageio.imwrite(out_path, cropped)


def find_all_trees(
    segmentid_dir: str,
    color_dir: str,
    output_dir: str,
    tree_attributes_dict: dict,
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
            semantic_face_path=semantic_path,
            image_number=f"{image_number}{orientation}",
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
        if len(parts) < 2:
            continue
        if parts[2] not in classes:
            classes.append(parts[2])
            os.makedirs(os.path.join(output_dataset_dir, parts[2]), exist_ok=True)
        if only_copy:
            shutil.copy2(
                os.path.join(input_trees_dir, tree),
                os.path.join(os.path.join(output_dataset_dir, parts[2]), tree),
            )
        else:
            shutil.move(
                os.path.join(input_trees_dir, tree),
                os.path.join(os.path.join(output_dataset_dir, parts[2]), tree),
            )
    print(f"Created dataset with {len(classes)} classes in {output_dataset_dir}")
