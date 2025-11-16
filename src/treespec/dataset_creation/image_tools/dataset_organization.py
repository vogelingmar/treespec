"""Functions to organize datasets based on tree attributes (i.e. "bark") and species."""

from pathlib import Path
import os
import shutil


def classify_trees_by_species(input_attribute_patches_dir_path: Path, output_attribute_dataset_dir_path: Path) -> None:
    r"""Creates a dataset from the extracted tree images based on their names.

    Args:
        input_attribute_patches_dir_path: Path to directory where the pictures for the dataset are stored.
        output_attribute_dataset_dir_path: Path to directory where the sorted dataset will be created.
    """
    classes = []
    copy = shutil.move if input_attribute_patches_dir_path == output_attribute_dataset_dir_path else shutil.copy2
    for tree_patch in os.listdir(input_attribute_patches_dir_path):
        filename = os.path.splitext(tree_patch)[0]
        parts = filename.split("_")
        if len(parts) < 5:
            continue
        species = parts[5]
        if species not in classes:
            classes.append(species)
            os.makedirs(os.path.join(output_attribute_dataset_dir_path, species), exist_ok=True)
        copy(
            os.path.join(input_attribute_patches_dir_path, tree_patch),
            os.path.join(output_attribute_dataset_dir_path, species, tree_patch),
        )
    print(f"Created dataset with {len(classes)} classes in {output_attribute_dataset_dir_path}")


def organize_datasets(input_tree_patches_dir_path: Path, output_datasets_dir_path: Path, tree_attributes: list) -> None:
    r"""Organizes datasets according to portrayed tree attribute and species.

    Args:
        input_trees_patches_dir_path: Path to directory where the directories containing pictures
            for each attribute dataset are stored.
        output_dataset_dir_path: Path to directory where the sorted datasets will be created.
        tree_attributes: List of tree attributes for which datasets should be created.
    """
    for attribute in tree_attributes:
        input_attribute_patches_dir_path = Path(os.path.join(input_tree_patches_dir_path, attribute))
        output_attribute_dataset_dir_path = Path(os.path.join(output_datasets_dir_path, attribute))
        os.makedirs(output_attribute_dataset_dir_path, exist_ok=True)

        classify_trees_by_species(input_attribute_patches_dir_path, output_attribute_dataset_dir_path)
