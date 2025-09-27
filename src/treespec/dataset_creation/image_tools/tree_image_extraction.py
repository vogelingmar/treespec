"""Functions for extracting tree and bark images from segment ID, color, and semantic image faces."""

import os
from pathlib import Path
from typing import Optional
import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import Counter


def determine_output_path(tree_id: int, tree_inventory_dict: dict, output_datasets_dir_path: Path, image_id: str, tree_attribute: str) -> Path:
    r"""
    Determines the output path for a tree image based on its ID and species.
    
    Args:
        tree_id: The ID of the tree.
        tree_inventory_dict: A dictionary containing tree inventory data.
        output_datasets_dir_path: The base directory for output datasets.
        image_id: The ID of the image being processed.
        tree_attribute: The attribute of the tree (e.g., 'tree', 'bark').
        
    Returns:
        The full path where the image should be saved.
    """

    #TODO: refactor code to be more elegant (code duplication)
    if tree_id in tree_inventory_dict.keys():
        tree_species = tree_inventory_dict[tree_id]["BAUMART"]
        tree_attribute_dataset_dir_path = os.path.join(output_datasets_dir_path, tree_attribute)
        return os.path.join(tree_attribute_dataset_dir_path, f"{tree_id}_{image_id}_{tree_species}.png")
    elif str(tree_id) in tree_inventory_dict.keys():
        tree_species = tree_inventory_dict[str(tree_id)]["BAUMART"]
        tree_attribute_dataset_dir_path = os.path.join(output_datasets_dir_path, tree_attribute)
        return os.path.join(tree_attribute_dataset_dir_path, f"{tree_id}_{image_id}_{tree_species}.png")
    else:
        tree_species = "unknown"
        unkown_trees_dir_path = os.path.join(output_datasets_dir_path, f"unkown_{tree_attribute}s")
        os.makedirs(unkown_trees_dir_path, exist_ok=True)
        return os.path.join(unkown_trees_dir_path, f"{tree_id}_{image_id}_{tree_species}.png")


def extract_masked_patches_and_bounds(mask: np.ndarray, color_face: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]]:
    r"""Extracts and filters the relevant zoomed and cropped patches from the color face based on the provided mask.
    
    Args:
        mask: A binary mask indicating the area of interest.
        color_face: The color image from which to extract patches.
        
    Returns:
        A tuple containing:
            - zoomed_cropped_color_patch: The cropped color patch with the mask applied,
            - zoomed_color_patch: The zoomed color patch without the mask,
            - zoomed_mask: The zoomed mask,
            - patch_bound_y0: The starting y-coordinate of the patch,
            - patch_bound_y1: The ending y-coordinate of the patch,
            - patch_bound_x0: The starting x-coordinate of the patch,
            - patch_bound_x1: The ending x-coordinate of the patch,
        or None if the patch is too small or too wide."""

    patch_coords = np.argwhere(mask)
    if patch_coords.size < 200 * 200 or patch_coords.shape[0] < patch_coords.shape[1]:  # filter small and wide images
        return

    patch_bound_y0, patch_bound_x0 = patch_coords.min(axis=0)
    patch_bound_y1, patch_bound_x1 = patch_coords.max(axis=0) + 1  # +1 for slicing

    zoomed_color_patch = color_face[patch_bound_y0:patch_bound_y1, patch_bound_x0:patch_bound_x1]
    zoomed_mask = mask[patch_bound_y0:patch_bound_y1, patch_bound_x0:patch_bound_x1]  # gives cropped mask of whole tree

    zoomed_cropped_color_patch = zoomed_color_patch * zoomed_mask[..., None]

    return (
        zoomed_cropped_color_patch,
        zoomed_color_patch,
        zoomed_mask,
        patch_bound_y0,
        patch_bound_y1,
        patch_bound_x0,
        patch_bound_x1,
    )


def determine_colored_area(cropped_color_face: np.ndarray) -> float:
    r"""Determines the proportion of colored (non-black) area in the zoomed and cropped color face for filtering.
    Args:
        cropped_color_face: The cropped color image of the tree or bark patch.

    Returns:
        The proportion of colored pixels in the cropped color face.
    """
    number_of_colored_pixels = np.count_nonzero(np.any(cropped_color_face != 0, axis=-1))
    number_of_total_pixels = cropped_color_face.shape[0] * cropped_color_face.shape[1]
    return number_of_colored_pixels / number_of_total_pixels


class TreeExtractionStatus(Enum):
    SUCCESS = auto()
    TREE_TOO_SMALL_OR_WIDE = auto()
    BARK_TOO_SMALL_OR_WIDE = auto()
    BARK_TOO_SPARSE = auto()
    UNKNOWN_SPECIES = auto()

@dataclass
class ExtractionMetrics:
    status_counts: Counter = field(default_factory=Counter)
    total_trees_detected: int = 0

    def update(self, status: TreeExtractionStatus):
        self.status_counts[status] += 1

    def print_summary(self, output_path: Path):
        print(f"\n Extraction Summary for output path: {output_path}")
        print(f"--------------------------------------------")
        print(f"Total tree IDs detected:     {self.total_trees_detected}")
        for status in TreeExtractionStatus:
            print(f"{status.name.replace('_', ' ').title()}: {self.status_counts[status]}")


def create_dataset_images(
    color_face: np.ndarray,
    segmentid_face: np.ndarray,
    semanticclass_face: np.ndarray,
    tree_inventory_dict: dict,
    output_tree_patch_dir_path: Path,
    image_id: str,
    tree_id: int,
    tree_attributes: list,
) -> Optional[TreeExtractionStatus]:
    r"""Creates and saves dataset images for each tree attribute given in tree_attributes.
    Returns a status enum for metrics collection.
    """
    segmentid_mask = segmentid_face == tree_id  # binary mask for the current tree id
    tree_patches_and_bounds = extract_masked_patches_and_bounds(segmentid_mask, color_face)
    if tree_patches_and_bounds is None:
        return TreeExtractionStatus.TREE_TOO_SMALL_OR_WIDE

    (
        zoomed_cropped_tree_patch,
        zoomed_tree_patch,
        zoomed_id_mask,
        tree_bound_y0,
        tree_bound_y1,
        tree_bound_x0,
        tree_bound_x1,
    ) = tree_patches_and_bounds

    semanticclass_zoomed = semanticclass_face[tree_bound_y0:tree_bound_y1, tree_bound_x0:tree_bound_x1]

    zoomed_bark_mask = (semanticclass_zoomed == 1).astype(
        np.uint8
    ) & zoomed_id_mask  # binary mask for bark only within the current tree id
    bark_patches = extract_masked_patches_and_bounds(zoomed_bark_mask, zoomed_tree_patch)
    if bark_patches is None:
        return TreeExtractionStatus.BARK_TOO_SMALL_OR_WIDE

    zoomed_cropped_bark_patch, zoomed_bark_patch, zoomed_bark_mask, _, _, _, _ = bark_patches

    if determine_colored_area(zoomed_cropped_bark_patch) < 0.5:
        return TreeExtractionStatus.BARK_TOO_SPARSE

    for attribute in tree_attributes:
        output_image = None
        match attribute:
            case "tree":
                output_image = zoomed_tree_patch
            case "tree_crop":
                output_image = zoomed_cropped_tree_patch
            case "bark":
                output_image = zoomed_bark_patch
            case "bark_crop":
                output_image = zoomed_cropped_bark_patch
        os.makedirs(os.path.join(output_tree_patch_dir_path, attribute), exist_ok=True)
        output_path = determine_output_path(tree_id, tree_inventory_dict, output_tree_patch_dir_path, image_id, attribute)
        imageio.imwrite(output_path, output_image)
        # If any output_path ends with unknown, return UNKNOWN_SPECIES
        if os.path.basename(output_path).startswith(f"{tree_id}_{image_id}_unknown"):
            return TreeExtractionStatus.UNKNOWN_SPECIES
    return TreeExtractionStatus.SUCCESS


def extract_tree_images(
    color_face_path: Path,
    segmentid_face_path: Path,
    semanticclass_face_path: Path,
    output_tree_patches_dir_path: Path,
    tree_inventory_dict: dict,
    image_id: str,
    tree_attributes: list,
    metrics: ExtractionMetrics,
) -> None:
    r"""Extracts tree/bark images from corresponding segmentid, color and semanticclass images.
    Updates metrics for each tree processed.
    """
    color_face = imageio.imread(color_face_path)
    segmentid_face = imageio.imread(segmentid_face_path)
    semanticclass_face = imageio.imread(semanticclass_face_path)

    # Resize segmentid_face to match color_face resolution using nearest-neighbor interpolation
    if segmentid_face.shape[:2] != color_face.shape[:2]:
        segmentid_face = resize(
            segmentid_face, color_face.shape[:2], order=0, preserve_range=True, anti_aliasing=False  # nearest-neighbor
        ).astype(segmentid_face.dtype)
    if semanticclass_face.shape[:2] != color_face.shape[:2]:
        semanticclass_face = resize(
            semanticclass_face,
            color_face.shape[:2],
            order=0,  # nearest-neighbor
            preserve_range=True,
            anti_aliasing=False,
        ).astype(semanticclass_face.dtype)

    tree_ids = np.unique(segmentid_face)
    for tree_id in tree_ids:
        if tree_id not in (0, 1):  # skip background and ground
            metrics.total_trees_detected += 1
            status = create_dataset_images(
                color_face=color_face,
                segmentid_face=segmentid_face,
                semanticclass_face=semanticclass_face,
                tree_inventory_dict=tree_inventory_dict,
                output_tree_patch_dir_path=output_tree_patches_dir_path,
                image_id=image_id,
                tree_id=tree_id,
                tree_attributes=tree_attributes,
            )
            if status:
                metrics.update(status)


def find_all_trees(
    input_color_faces_dir_path: Path,
    input_color_faces_filetype: str,
    input_segmentid_faces_dir_path: Path,
    input_segmentid_faces_filetype: str,
    input_semanticclass_faces_dir_path: Path,
    input_semanticclass_faces_filetype: str,
    output_dataset_dir_path: Path,
    tree_inventory_dict: dict,
    run_number: int,
    date: str,
    tree_attributes: list,
) -> None:
    r"""Finds and extracts tree images from segment ID, color, and semantic images from the specified directories by matching their IDs.
    Uses ExtractionMetrics to collect and print summary statistics.
    """
    metrics = ExtractionMetrics()
    os.makedirs(output_dataset_dir_path, exist_ok=True)
    for segmentid_face in os.listdir(input_segmentid_faces_dir_path):
        if segmentid_face.endswith(f".{input_segmentid_faces_filetype}"):
            filename = os.path.splitext(segmentid_face)[0]
            parts = filename.split("_")
            if len(parts) < 3:
                continue
            image_number = parts[0]
            orientation = parts[2]
            color_face_path = os.path.join(
                input_color_faces_dir_path, f"{image_number}_rgb_{orientation}.{input_color_faces_filetype}"
            )
            semanticclass_face_path = os.path.join(
                input_semanticclass_faces_dir_path,
                f"{image_number}_semanticclass_{orientation}.{input_semanticclass_faces_filetype}",
            )

            segmentid_face_path = os.path.join(input_segmentid_faces_dir_path, segmentid_face)
            extract_tree_images(
                segmentid_face_path=segmentid_face_path,
                color_face_path=color_face_path,
                output_tree_patches_dir_path=output_dataset_dir_path,
                tree_inventory_dict=tree_inventory_dict,
                semanticclass_face_path=semanticclass_face_path,
                image_id=f"{date}_{run_number}_{image_number}_{orientation}",
                tree_attributes=tree_attributes,
                metrics=metrics,
            )

    metrics.print_summary(output_dataset_dir_path)
