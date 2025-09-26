import os
import shutil
from treespec.dataset_creation.image_tools.dataset_organization import organize_datasets
from treespec.dataset_creation.inventory_tools.inventory_convertion import create_dictionary_from_shapefile

dataset_creation_mock_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")


def test_organize_datasets():
    """Tests the organize_datasets function."""
    input_trees_patches_dir_path = os.path.join(dataset_creation_mock_dir_path, "datasets", "dataset_unsorted")
    output_dataset_dir_path = os.path.join(dataset_creation_mock_dir_path, "temp", "dataset_sorted")
    tree_attributes = ["tree_crop"]

    shutil.rmtree(output_dataset_dir_path, ignore_errors=True)
    os.makedirs(output_dataset_dir_path, exist_ok=True)

    organize_datasets(
        input_tree_patches_dir_path=input_trees_patches_dir_path,
        output_datasets_dir_path=output_dataset_dir_path,
        tree_attributes=tree_attributes,
    )

    attribute_path = os.path.join(output_dataset_dir_path, "tree_crop")

    assert os.path.exists(attribute_path)
    dataset = os.listdir(attribute_path)
    assert len(dataset) == 3
    for dir in dataset:
        dir_path = os.path.join(attribute_path, dir)
        for file in os.listdir(dir_path):
            assert file.endswith(".png")
            parts = file.split("_")
            assert len(parts) == 6
            assert parts[0].isdigit()

    shutil.rmtree(output_dataset_dir_path, ignore_errors=True)
