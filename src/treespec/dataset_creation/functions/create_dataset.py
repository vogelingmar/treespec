"""Creates an image dataset from color, segmentid and semanticclass images/ panoramas."""

import os
from pathlib import Path

from treespec.dataset_creation.image_tools import pre_processing, tree_image_extraction, dataset_organization
from treespec.dataset_creation.inventory_tools.inventory_convertion import create_dictionary_from_shapefile
from treespec.dataset_creation.functions import match_inventories


def pre_process(
    input_color_images_dir_path: Path,
    input_color_image_filetype: str,
    input_color_images_format: str,
    input_segmentid_images_dir_path: Path,
    input_segmentid_image_filetype: str,
    input_semanticclass_images_dir_path: Path,
    input_semanticclass_image_filetype: str,
    run_number: int,
    processed_color_images_path: Path,
    processed_color_image_filetype: str,
    processed_segmentid_images_path: Path,
    processed_segmentid_image_filetype: str,
    processed_semanticclass_images_path: Path,
    processed_semanticclass_image_filetype: str,
)-> None:
    r"""Pre-processes the input images by extracting faces from panoramas or selecting rectangle images depending on the input_color_images_format.

    Args:
        input_color_images_dir_path: Path to the directory containing the input color images or panoramas.
        input_color_image_filetype: Filetype of the input color images or panoramas (e.g. "jpg", "png", "tif").
        input_color_images_format: Format of the input color images, either "pano" for panoramas or "rectangle" for rectangle images.
        input_segmentid_images_dir_path: Path to the directory containing the input segmentid panoramas.
        input_segmentid_image_filetype: Filetype of the input segmentid panoramas (e.g. "jpg", "png", "tif").
        input_semanticclass_images_dir_path: Path to the directory containing the input semanticclass panoramas.
        input_semanticclass_image_filetype: Filetype of the input semanticclass panoramas (e.g. "jpg", "png", "tif").
        run_number: Run number of the input images.
        processed_color_images_path: Path to the directory where the processed color images will be saved.
        processed_color_image_filetype: Filetype of the processed color images (e.g. "jpg", "png", "tif").
        processed_segmentid_images_path: Path to the directory where the processed segmentid images will be saved.
        processed_segmentid_image_filetype: Filetype of the processed segmentid images (e.g. "jpg", "png", "tif").
        processed_semanticclass_images_path: Path to the directory where the processed semanticclass images will be saved.
        processed_semanticclass_image_filetype: Filetype of the processed semanticclass images (e.g. "jpg", "png", "tif").
    """

    if input_color_images_format == "pano":
        apply_center_crop = False
        pre_processing.extract_pano_faces(
            input_panos_dir_path=input_color_images_dir_path,
            output_faces_dir_path=processed_color_images_path,
            input_pano_filetype=input_color_image_filetype,
            output_face_filetype=processed_color_image_filetype,
            run_number=run_number,
            apply_center_zoom=apply_center_crop,
        )
    else:
        apply_center_crop = True
        pre_processing.select_rectangle_images(
            input_rectangle_images_dir_path=input_color_images_dir_path,
            output_image_faces_dir_path=processed_color_images_path,
            input_rectangle_image_filetype=input_color_image_filetype,
            output_color_image_filetype=processed_color_image_filetype,
            run_number=run_number,
        )

    pre_processing.extract_pano_faces(
        input_panos_dir_path=input_segmentid_images_dir_path,
        output_faces_dir_path=processed_segmentid_images_path,
        input_pano_filetype=input_segmentid_image_filetype,
        output_face_filetype=processed_segmentid_image_filetype,
        run_number=run_number,
        apply_center_zoom=apply_center_crop,
        name_filter="segmentid",
    )
    pre_processing.extract_pano_faces(
        input_panos_dir_path=input_semanticclass_images_dir_path,
        output_faces_dir_path=processed_semanticclass_images_path,
        input_pano_filetype=input_semanticclass_image_filetype,
        output_face_filetype=processed_semanticclass_image_filetype,
        run_number=run_number,
        apply_center_zoom=apply_center_crop,
        name_filter="semanticclass",
    )


def create_dataset(
    input_color_images_dir_path: Path,
    input_color_image_filetype: str,
    input_color_images_format: str,
    input_segmentid_images_dir_path: Path,
    input_segmentid_image_filetype: str,
    input_semanticclass_images_dir_path: Path,
    input_semanticclass_image_filetype: str,
    pre_processed: bool,
    date: str,
    run_number: int,
    processed_color_images_path: Path,
    processed_color_image_filetype: str,
    processed_segmentid_images_path: Path,
    processed_segmentid_image_filetype: str,
    processed_semanticclass_images_path: Path,
    processed_semanticclass_image_filetype: str,
    output_dataset_dir_path: Path,
    input_tree_inventory_path: Path,
    tree_attributes: list,
)-> None:
    r"""Creates a dataset from one run.
    Args:
        input_color_images_dir_path: Path to the directory containing the input color images or panoramas.
        input_color_image_filetype: Filetype of the input color images or panoramas (e.g. "jpg", "png", "tif").
        input_color_images_format: Format of the input color images, either "pano" for panoramas or "rectangle" for rectangle images.
        input_segmentid_images_dir_path: Path to the directory containing the input segmentid panoramas.
        input_segmentid_image_filetype: Filetype of the input segmentid panoramas (e.g. "jpg", "png", "tif").
        input_semanticclass_images_dir_path: Path to the directory containing the input semanticclass panoramas.
        input_semanticclass_image_filetype: Filetype of the input semanticclass panoramas (e.g. "jpg", "png", "tif").
        pre_processed: Boolean indicating whether the input images have already been pre-processed.
        date: Date of the input images in the format "YYYY-MM-DD".
        run_number: Run number of the input images.
        processed_color_images_path: Path to the directory where the processed color images will be saved.
        processed_color_image_filetype: Filetype of the processed color images (e.g. "jpg", "png", "tif").
        processed_segmentid_images_path: Path to the directory where the processed segmentid images will be saved.
        processed_segmentid_image_filetype: Filetype of the processed segmentid images (e.g. "jpg", "png", "tif").
        processed_semanticclass_images_path: Path to the directory where the processed semanticclass images will be saved.
        processed_semanticclass_image_filetype: Filetype of the processed semanticclass images (e.g. "jpg", "png", "tif").
        output_dataset_dir_path: Path to the directory where the output dataset will be saved.
        input_tree_inventory_path: Path to the tree inventory file.
        tree_attributes: List of tree attributes to be included in the dataset (e.g. ["tree", "tree_crop", "bark", "bark_crop"]).
    """

    if not pre_processed:
        pre_process(
            input_color_images_dir_path=input_color_images_dir_path,
            input_color_image_filetype=input_color_image_filetype,
            input_color_images_format=input_color_images_format,
            input_segmentid_images_dir_path=input_segmentid_images_dir_path,
            input_segmentid_image_filetype=input_segmentid_image_filetype,
            input_semanticclass_images_dir_path=input_semanticclass_images_dir_path,
            input_semanticclass_image_filetype=input_semanticclass_image_filetype,
            run_number=run_number,
            processed_color_images_path=processed_color_images_path,
            processed_color_image_filetype=processed_color_image_filetype,
            processed_segmentid_images_path=processed_segmentid_images_path,
            processed_segmentid_image_filetype=processed_segmentid_image_filetype,
            processed_semanticclass_images_path=processed_semanticclass_images_path,
            processed_semanticclass_image_filetype=processed_semanticclass_image_filetype,
        )

    tree_inventory_dict = create_dictionary_from_shapefile(input_tree_inventory_path)

    tree_image_extraction.find_all_trees(
        input_color_faces_dir_path=processed_color_images_path,
        input_color_faces_filetype=processed_color_image_filetype,
        input_segmentid_faces_dir_path=processed_segmentid_images_path,
        input_segmentid_faces_filetype=processed_segmentid_image_filetype,
        input_semanticclass_faces_dir_path=processed_semanticclass_images_path,
        input_semanticclass_faces_filetype=processed_semanticclass_image_filetype,
        output_dataset_dir_path=output_dataset_dir_path,
        tree_inventory_dict=tree_inventory_dict,
        run_number=run_number,
        date=date,
        tree_attributes=tree_attributes,
    )

    dataset_organization.organize_datasets(
        input_tree_patches_dir_path=output_dataset_dir_path,
        output_datasets_dir_path=output_dataset_dir_path,
        tree_attributes=tree_attributes,
    )


def create_simple_dataset(
    input_color_images_format: str,
    date: str,
    groundtruth_tree_inventory_path: Path,
    input_dir_path: Path,
    processed_dir_path: Path,
    pre_processed: bool,
    output_dataset_dir_path: Path,
    run_numbers: list,
) -> None:
    r"""Creates a dataset from many runs of one date.
    Args:
        input_color_images_format: Format of the input color images, either "pano" for panoramas or "rectangle" for rectangle images.
        date: Date of the input images in the format "YYYY-MM-DD".
        groundtruth_tree_inventory_path: Path to the ground truth tree inventory file.
        input_dir_path: Path to the directory containing the input images and inventories.
        processed_dir_path: Path to the directory where the processed images will be saved.
        pre_processed: Boolean indicating whether the input images have already been pre-processed and with that lie in the correct directory.
        output_dataset_dir_path: Path to the directory where the output dataset will be saved.
        run_numbers: List of run numbers to be processed.
    """

    input_color_images_dir_path = (
        os.path.join(input_dir_path, "panos")
        if input_color_images_format == "pano"
        else os.path.join(input_dir_path, "images")
    )
    input_color_image_filetype = "jpg"
    input_segmentid_image_filetype = "tif"
    input_semanticclass_image_filetype = "tif"
    processed_color_image_filetype = "png"
    processed_segmentid_image_filetype = "png"
    processed_semanticclass_image_filetype = "png"

    for run_number in run_numbers:
        input_segmentid_images_dir_path = os.path.join(input_dir_path, "panos", f"rend{run_number}")
        input_semanticclass_images_dir_path = os.path.join(input_dir_path, "panos", f"rend{run_number}")
        processed_color_images_path = os.path.join(
            processed_dir_path, date, f"run{run_number}", f"color_faces_{date}_{run_number}"
        )
        processed_segmentid_images_path = os.path.join(
            processed_dir_path, date, f"run{run_number}", f"segmentid_faces_{date}_{run_number}"
        )
        processed_semanticclass_images_path = os.path.join(
            processed_dir_path, date, f"run{run_number}", f"semanticclass_faces_{date}_{run_number}"
        )
        tree_attributes = ["tree", "tree_crop", "bark", "bark_crop"]

        predicted_tree_inventory_path = os.path.join(
            input_dir_path, "inventory", f"run{run_number}", f"inventory{run_number}"
        )
        matched_tree_inventory_output_path = os.path.join(
            processed_dir_path, date, f"run{run_number}", f"matched_inventory_{date}_{run_number}", "matched_inventory"
        )
        match_inventories.match(
            predicted_inventory_path=predicted_tree_inventory_path,
            groundtruth_inventory_path=groundtruth_tree_inventory_path,
            output_inventory_path=matched_tree_inventory_output_path,
            use_dbh_filter=False,
        )
        input_tree_inventory_path = matched_tree_inventory_output_path

        create_dataset(
            input_color_images_dir_path=input_color_images_dir_path,
            input_color_image_filetype=input_color_image_filetype,
            input_color_images_format=input_color_images_format,
            input_segmentid_images_dir_path=input_segmentid_images_dir_path,
            input_segmentid_image_filetype=input_segmentid_image_filetype,
            input_semanticclass_images_dir_path=input_semanticclass_images_dir_path,
            input_semanticclass_image_filetype=input_semanticclass_image_filetype,
            pre_processed=pre_processed,
            date=date,
            run_number=run_number,
            processed_color_images_path=processed_color_images_path,
            processed_color_image_filetype=processed_color_image_filetype,
            processed_segmentid_images_path=processed_segmentid_images_path,
            processed_segmentid_image_filetype=processed_segmentid_image_filetype,
            processed_semanticclass_images_path=processed_semanticclass_images_path,
            processed_semanticclass_image_filetype=processed_semanticclass_image_filetype,
            output_dataset_dir_path=output_dataset_dir_path,
            input_tree_inventory_path=input_tree_inventory_path,
            tree_attributes=tree_attributes,
        )


def create_big_scale_dataset(input_dir_path: Path, output_dir_path: Path, dates_and_runs: dict) -> None:
    r"""Creates a large dataset from many different dates and runs.
    Args:
        input_dir_path: Path to the directory containing the input images and inventories.
        output_dir_path: Path to the directory where the output dataset will be saved.
        dates_and_runs: Dictionary with dates as keys and lists of run numbers as values."""
    
    date_dirs = os.listdir(input_dir_path)
    for date_dir in date_dirs:
        if date_dir in dates_and_runs.keys():
            run_numbers = dates_and_runs[date_dir]
            input_color_images_format = "pano"
            date = date_dir
            groundtruth_tree_inventory_path = os.path.join(input_dir_path, "groundtruth_inventory", "inventory")
            processed_dir_path = os.path.join(output_dir_path, "pre_processing")
            pre_processed = False
            output_dataset_dir_path = os.path.join(output_dir_path, "datasets")
            input_dir_path = os.path.join(input_dir_path, date_dir)

            create_simple_dataset(
                input_color_images_format=input_color_images_format,
                date=date,
                groundtruth_tree_inventory_path=groundtruth_tree_inventory_path,
                input_dir_path=input_dir_path,
                processed_dir_path=processed_dir_path,
                pre_processed=pre_processed,
                output_dataset_dir_path=output_dataset_dir_path,
                run_numbers=run_numbers,
            )
