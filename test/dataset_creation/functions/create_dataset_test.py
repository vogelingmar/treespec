import os
import shutil
from treespec.dataset_creation.functions.create_dataset import (
    create_dataset,
    create_simple_dataset,
    create_big_scale_dataset,
)


def test_create_dataset():
    dataset_creation_mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")
    shutil.rmtree(os.path.join(dataset_creation_mock_path, "temp"), ignore_errors=True)

    input_color_images_dir_path = os.path.join(dataset_creation_mock_path, "images", "pano_color_images")
    input_color_image_filetype = "jpg"
    input_color_images_format = "pano"
    input_segmentid_images_dir_path = os.path.join(dataset_creation_mock_path, "images", "pano_segmentid_images")
    input_segmentid_image_filetype = "tif"
    input_semanticclass_images_dir_path = os.path.join(
        dataset_creation_mock_path, "images", "pano_semanticclass_images"
    )
    input_semanticclass_image_filetype = "tif"
    pre_processed = False
    date = "2025-09-10"
    run_number = 70
    processed_color_images_path = os.path.join(
        dataset_creation_mock_path, "temp", "processed", "processed_color_images"
    )
    processed_color_image_filetype = "png"
    processed_segmentid_images_path = os.path.join(
        dataset_creation_mock_path, "temp", "processed", "processed_segmentid_images"
    )
    processed_segmentid_image_filetype = "png"
    processed_semanticclass_images_path = os.path.join(
        dataset_creation_mock_path, "temp", "processed", "processed_semanticclass_images"
    )
    processed_semanticclass_image_filetype = "png"
    output_dataset_dir_path = os.path.join(dataset_creation_mock_path, "temp", "dataset", "dataset")
    input_tree_inventory_path = os.path.join(
        dataset_creation_mock_path, "inventories", "inventory_matched", "matched_output"
    )
    tree_attributes = ["tree", "tree_crop", "bark", "bark_crop"]

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
        input_tree_inventory_path=input_tree_inventory_path,
        output_dataset_dir_path=output_dataset_dir_path,
        tree_attributes=tree_attributes,
    )

    assert len(os.listdir(processed_color_images_path)) > 0, "Processed color images directory is empty!"
    assert len(os.listdir(processed_segmentid_images_path)) > 0, "Processed segmentid images directory is empty!"
    assert (
        len(os.listdir(processed_semanticclass_images_path)) > 0
    ), "Processed semanticclass images directory is empty!"
    assert len(os.listdir(output_dataset_dir_path)) > 0, "Output dataset directory is empty!"
    number_of_found_trees = 0
    for tree_attribute in tree_attributes:
        attribute_dir = os.path.join(output_dataset_dir_path, tree_attribute)
        assert os.path.exists(attribute_dir), f"Directory for {tree_attribute} does not exist!"
        assert len(os.listdir(attribute_dir)) > 0, f"{tree_attribute} images directory is empty!"
        if number_of_found_trees == 0:
            number_of_found_trees = len(os.listdir(attribute_dir))
        else:
            assert (
                len(os.listdir(attribute_dir)) == number_of_found_trees
            ), f"Number of images in output datasets do not match!"

    shutil.rmtree(os.path.join(dataset_creation_mock_path, "temp"))


def test_create_simple_dataset():
    dataset_creation_mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")
    shutil.rmtree(os.path.join(dataset_creation_mock_path, "temp"), ignore_errors=True)

    input_color_images_format = "rectangle"
    groundtruth_tree_inventory_path = os.path.join(
        dataset_creation_mock_path, "inventories", "inventory_groundtruth", "cadastre_essen"
    )
    input_dir_path = os.path.join(dataset_creation_mock_path, "large_scale_dataset_creation_input", "2025-09-10")
    processed_dir_path = os.path.join(dataset_creation_mock_path, "temp", "simple_dataset", "processed")
    pre_processed = False
    output_dataset_dir_path = os.path.join(dataset_creation_mock_path, "temp", "simple_dataset", "dataset")
    run_numbers = [70]
    date = "2025-09-10"

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

    assert len(os.listdir(processed_dir_path)) > 0, "Processed directory is empty!"
    assert len(os.listdir(output_dataset_dir_path)) > 0, "Output dataset directory is empty!"
    processed_color_images_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", "color_faces_2025-09-10_70"
    )
    processed_segmentid_images_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", "segmentid_faces_2025-09-10_70"
    )
    processed_semanticclass_images_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", "semanticclass_faces_2025-09-10_70"
    )
    matched_inventory_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", "matched_inventory_2025-09-10_70", "matched_inventory"
    )

    assert len(os.listdir(processed_color_images_path)) > 0, "Processed color images directory is empty!"
    assert len(os.listdir(processed_segmentid_images_path)) > 0, "Processed segmentid images directory is empty!"
    assert (
        len(os.listdir(processed_semanticclass_images_path)) > 0
    ), "Processed semanticclass images directory is empty!"
    assert len(os.listdir(os.path.dirname(matched_inventory_path))) > 0, "Matched inventory directory is empty!"
    assert len(os.listdir(output_dataset_dir_path)) > 0, "Output dataset directory is empty!"
    tree_attributes = ["tree", "tree_crop", "bark", "bark_crop"]
    for tree_attribute in tree_attributes:
        attribute_dir = os.path.join(output_dataset_dir_path, tree_attribute)
        assert os.path.exists(attribute_dir), f"Directory for {tree_attribute} does not exist!"
        assert len(os.listdir(attribute_dir)) > 0, f"{tree_attribute} images directory is empty!"

    shutil.rmtree(os.path.join(dataset_creation_mock_path, "temp"))


def test_create_big_scale_dataset():
    dataset_creation_mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")
    shutil.rmtree(os.path.join(dataset_creation_mock_path, "temp"), ignore_errors=True)

    input_dir_path = os.path.join(dataset_creation_mock_path, "large_scale_dataset_creation_input")
    output_dir_path = os.path.join(dataset_creation_mock_path, "temp", "large_scale_dataset")
    dates_and_runs = {"2025-09-10": [70]}

    create_big_scale_dataset(
        input_dir_path=input_dir_path,
        output_dir_path=output_dir_path,
        dates_and_runs=dates_and_runs,
    )

    dataset_dir_path = os.path.join(output_dir_path, "datasets")
    processed_dir_path = os.path.join(output_dir_path, "pre_processing")
    date = dates_and_runs.keys().__iter__().__next__()
    run_numbers = dates_and_runs[date]

    assert len(os.listdir(output_dir_path)) > 0, "Processed directory is empty!"
    assert len(os.listdir(dataset_dir_path)) > 0, "Output dataset directory is empty!"
    assert len(os.listdir(processed_dir_path)) > 0, "Pre processing directory is empty!"
    processed_color_images_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", f"color_faces_2025-09-10_70"
    )
    processed_segmentid_images_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", f"segmentid_faces_2025-09-10_70"
    )
    processed_semanticclass_images_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", f"semanticclass_faces_2025-09-10_70"
    )
    matched_inventory_path = os.path.join(
        processed_dir_path, date, f"run{run_numbers[0]}", f"matched_inventory_2025-09-10_70", "matched_inventory"
    )

    assert len(os.listdir(processed_color_images_path)) > 0, "Processed color images directory is empty!"
    assert len(os.listdir(processed_segmentid_images_path)) > 0, "Processed segmentid images directory is empty!"
    assert (
        len(os.listdir(processed_semanticclass_images_path)) > 0
    ), "Processed semanticclass images directory is empty!"
    assert len(os.listdir(os.path.dirname(matched_inventory_path))) > 0, "Matched inventory directory is empty!"
    assert len(os.listdir(dataset_dir_path)) > 0, "Output dataset directory is empty!"
    for tree_attribute in ["tree", "tree_crop", "bark", "bark_crop"]:
        attribute_dir = os.path.join(dataset_dir_path, tree_attribute)
        assert os.path.exists(attribute_dir), f"Directory for {tree_attribute} does not exist!"
        assert len(os.listdir(attribute_dir)) > 0, f"{tree_attribute} images directory is empty!"

    shutil.rmtree(os.path.join(dataset_creation_mock_path, "temp"))
