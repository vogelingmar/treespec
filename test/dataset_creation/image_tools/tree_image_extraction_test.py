"""Tests for the tree_image_extraction module."""

import os
import shutil
from treespec.dataset_creation.image_tools.tree_image_extraction import find_all_trees
from treespec.dataset_creation.inventory_tools.inventory_convertion import create_dictionary_from_shapefile

dataset_creation_mock_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")


def test_find_all_trees():
    """Tests the find_all_trees function for both tree and bark covers."""
    input_color_faces_dir_path = os.path.join(dataset_creation_mock_dir_path, "faces", "face_color_images")
    input_color_faces_filetype = "png"
    input_segmentid_faces_dir_path = os.path.join(dataset_creation_mock_dir_path, "faces", "face_segmentid_images")
    input_segmentid_faces_filetype = "png"
    input_semanticclass_faces_dir_path = os.path.join(
        dataset_creation_mock_dir_path, "faces", "face_semanticclass_images"
    )
    input_semanticclass_faces_filetype = "png"
    output_dataset_dir_path = os.path.join(dataset_creation_mock_dir_path, "temp", "dataset")
    tree_inventory_dict = create_dictionary_from_shapefile(
        os.path.join(dataset_creation_mock_dir_path, "inventories", "inventory_matched", "matched_output")
    )
    run_number = 70
    date = "2025-09-15"
    tree_attributes = ["tree", "bark", "tree_crop", "bark_crop"]

    # test for trees
    shutil.rmtree(output_dataset_dir_path, ignore_errors=True)
    os.makedirs(output_dataset_dir_path, exist_ok=True)

    find_all_trees(
        input_color_faces_dir_path=input_color_faces_dir_path,
        input_color_faces_filetype=input_color_faces_filetype,
        input_segmentid_faces_dir_path=input_segmentid_faces_dir_path,
        input_segmentid_faces_filetype=input_segmentid_faces_filetype,
        input_semanticclass_faces_dir_path=input_semanticclass_faces_dir_path,
        input_semanticclass_faces_filetype=input_semanticclass_faces_filetype,
        output_dataset_dir_path=output_dataset_dir_path,
        tree_inventory_dict=tree_inventory_dict,
        run_number=run_number,
        date=date,
        tree_attributes=tree_attributes,
    )

    output = os.listdir(output_dataset_dir_path)
    assert len(output) > 0
    for attribute in output:
        for file in os.listdir(os.path.join(output_dataset_dir_path, attribute)):
            parts = file.split("_")
            assert len(parts) == 6
            assert parts[0].isdigit()

    shutil.rmtree(output_dataset_dir_path, ignore_errors=True)
