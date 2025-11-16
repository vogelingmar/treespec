"""Tests for the predict function."""

import shutil
import os

from torch import nn
from torchvision.models import googlenet, GoogLeNet_Weights  # type: ignore

from treespec.classification.functions.predict import predict_species, _inventurize_trees
from treespec.dataset_creation.inventory_tools.inventory_convertion import create_dictionary_from_shapefile
from treespec.classification.image_dataset import ImageDataset
from treespec.classification.classification_model import ClassificationModel

test_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def test_predict_species():

    model = googlenet
    model_weights = GoogLeNet_Weights.DEFAULT
    dataset_dir = os.path.join(test_path, "dataset_creation", "mock", "datasets", "dataset_sorted")
    tree_images_dir = os.path.join(test_path, "dataset_creation", "mock", "datasets", "dataset_unsorted", "tree_crop")
    input_inventory_path = os.path.join(
        test_path, "dataset_creation", "mock", "inventories", "inventory_matched", "matched_output"
    )
    output_inventory_path = os.path.join(
        test_path, "classification", "mock", "temp", "predicted_species_inventory", "pred_species_inventory"
    )
    trained_model_path = os.path.join(
        test_path, "classification", "mock", "trained_classification_models", "GoogLeNet_dataset_sorted_5_finetuned"
    )

    predict_species(
        model=model,
        model_weights=model_weights,
        dataset_dir_path=dataset_dir,
        tree_images_dir_path=tree_images_dir,
        input_inventory_path=input_inventory_path,
        output_inventory_path=output_inventory_path,
        trained_model_path=trained_model_path,
    )
    assert len(os.listdir(os.path.dirname(output_inventory_path))) > 0

    shutil.rmtree(os.path.dirname(output_inventory_path), ignore_errors=True)


def test_inventurize_trees():
    """Tests the match_predicted_tree_species function."""
    dataset_creation_mock_dir_path = os.path.join(test_path, "dataset_creation", "mock")
    tree_images_dir = os.path.join(dataset_creation_mock_dir_path, "datasets", "dataset_unsorted", "tree_crop")
    input_inventory_path = os.path.join(
        dataset_creation_mock_dir_path, "inventories", "inventory_matched", "matched_output"
    )
    output_inventory_path = os.path.join(
        dataset_creation_mock_dir_path, "temp", "predicted_species_inventory", "pred_species_inventory"
    )

    dataset = ImageDataset(
        dataset_dir_path=os.path.join(dataset_creation_mock_dir_path, "datasets", "dataset_sorted"),
        batch_size=3,
        num_workers=2,
        use_ids=True,
    )
    classification_model = ClassificationModel(
        model=googlenet,
        model_weights=GoogLeNet_Weights.DEFAULT,
        num_classes=3,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=0.001,
    )

    shutil.rmtree(os.path.dirname(output_inventory_path), ignore_errors=True)

    _inventurize_trees(
        input_tree_images_dir_path=tree_images_dir,
        input_inventory_path=input_inventory_path,
        output_inventory_path=output_inventory_path,
        classification_model=classification_model,
        class_names=dataset.classes,
        filetypes=[".jpg", ".png", ".jpeg"],
    )

    assert os.path.exists(output_inventory_path + ".shp")
    assert os.path.exists(output_inventory_path + ".dbf")
    assert os.path.exists(output_inventory_path + ".shx")

    inventory = create_dictionary_from_shapefile(output_inventory_path)

    assert len(inventory) > 0

    for tree in inventory.values():
        assert "pred_speci" in tree.keys()
        assert tree["pred_speci"] is not None or tree["pred_speci"] != ""

    shutil.rmtree(os.path.dirname(output_inventory_path), ignore_errors=True)
