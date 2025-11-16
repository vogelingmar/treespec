"""Prediction function for tree species classification."""

import os
from typing import Callable
from pathlib import Path

import torch
from torch import nn
from torchvision.io import decode_image  # type: ignore
from torchvision.models._api import WeightsEnum  # type: ignore

from treespec.classification.classification_model import ClassificationModel
from treespec.classification.image_dataset import ImageDataset
from treespec.dataset_creation.inventory_tools.inventory_convertion import (
    create_dictionary_from_shapefile,
    create_shapefile_from_dictionary,
)


def _predict(img_path: Path, classification_model: ClassificationModel) -> dict:
    r"""
    The predict function using the classification model.

    Args:
        img_path: The path to the image to be predicted.
        classification_model: The classification model instance used for prediction.

    Returns:
        A dictionary containing the predicted category and confidence score.
    """
    try:
        picture = decode_image(img_path)
    except Exception as e:
        raise ValueError(f"Could not read image: {e}") from e

    batch = classification_model.model_weights.transforms()(picture).unsqueeze(0)
    device = next(classification_model.model.parameters()).device
    batch = batch.to(device)
    class_id, score = classification_model.predict_step(batch, 0)
    return {"category": class_id, "score": score}


def _inventurize_trees(  # pylint: disable=too-many-arguments, too-many-positional-arguments, dangerous-default-value, too-many-locals, too-many-branches
    input_tree_images_dir_path: Path,
    input_inventory_path: Path,
    output_inventory_path: Path,
    classification_model: ClassificationModel,
    class_names: list,
    filetypes: list = [".jpg", ".png", ".jpeg"],
) -> None:
    r"""Predicts tree species on input trees and exports predictions as an inventory shapefile.

    Args:
        input_tree_images_dir_path: Directory containing images of trees to classify.
        input_inventory_path: Path to the input inventory shapefile.
        output_inventory_path: Path to save the updated inventory shapefile with predicted species.
        classification_model: Instance of the ClassificationModel used for prediction.
        class_names: List of class names corresponding to the model's output classes.
        filetypes: List of acceptable image file extensions.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classification_model = classification_model.to(device)
    classification_model.eval()  # Set the model to evaluation mode

    trees = create_dictionary_from_shapefile(input_inventory_path)

    for tree_name in os.listdir(input_tree_images_dir_path):
        image_path = Path(os.path.join(input_tree_images_dir_path, tree_name))

        for filetype in filetypes:
            if os.path.isdir(image_path) or not str(image_path).lower().endswith(filetype):
                continue

        prediction = _predict(image_path, classification_model)
        predicted_class_id = prediction["category"]
        predicted_class = class_names[predicted_class_id]

        parts = os.path.splitext(tree_name)[0].split("_")

        tree_id = int(parts[0])
        image_id = str(parts[1])

        if tree_id in trees:
            if "pred_species" in trees[tree_id]:
                trees[tree_id][f"pred_sp_{image_id}"] = predicted_class
            else:
                trees[tree_id]["pred_species"] = predicted_class
        else:
            print(f"Tree ID {tree_id} not found in the matched cadastre data. Skipping.")

    for tree in trees.values():
        number_of_votes = 0
        votes: dict[str, int] = {}
        attributes = tree.keys()
        for attribute in attributes:
            if attribute.startswith("pred_sp"):
                number_of_votes += 1
                species = tree[attribute]
                votes[species] = votes.get(species, 0) + 1

        # Find if any species has majority
        majority_species = None
        for species, count in votes.items():
            if count > number_of_votes / 2:
                majority_species = species
                break

        if majority_species:
            tree["pred_species"] = majority_species
        else:
            tree["pred_species"] = "uncertain"
    create_shapefile_from_dictionary(trees, output_inventory_path)


def predict_species(  # pylint: disable=too-many-arguments, too-many-positional-arguments, dangerous-default-value, too-many-locals
    model: Callable,
    model_weights: WeightsEnum,
    dataset_dir_path: Path,
    tree_images_dir_path: Path,
    input_inventory_path: Path,
    output_inventory_path: Path,
    trained_model_path: Path,
    filetypes: list = [".jpg", ".png", ".jpeg"],
) -> None:
    """
    Predicts tree species from images and updates the inventory shapefile with the predictions.

    This function initializes the classification model and dataset, and then calls the `match_predicted_tree_species`
    function to perform the predictions and update the inventory.

    Args:
        model: A callable that returns the model architecture.
        model_weights: Pretrained weights for the model.
        dataset_dir: Path to the directory containing the dataset used for training the model.
        tree_images_dir: Path to the directory containing images of trees to classify.
        input_inventory_path: Path to the input inventory shapefile.
        output_inventory_path: Path to save the updated inventory shapefile with predicted species.
        trained_model_path: Path to the trained classification model checkpoint.
        filetypes: List of acceptable image file extensions.
    """

    dataset_instance = ImageDataset(
        dataset_dir_path=dataset_dir_path,
        batch_size=1,
        num_workers=0,
        use_ids=True,
    )
    default_transforms = model_weights.transforms()
    dataset_instance.setup(transform=default_transforms)

    class_names = dataset_instance.classes

    num_classes = len(class_names)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1, weight=dataset_instance.loss_weights())

    classification_model = ClassificationModel(
        model=model,
        model_weights=model_weights,
        num_classes=num_classes,
        loss_function=loss_function,
        learning_rate=0.001,
    )

    state_dict = torch.load(trained_model_path, map_location="cpu")

    for key in [
        "model.classifier.6.weight",
        "model.classifier.6.bias",
        "loss_function.weight",
    ]:
        if key in state_dict:
            del state_dict[key]

    classification_model.model.load_state_dict(state_dict, strict=False)

    _inventurize_trees(
        classification_model=classification_model,
        class_names=class_names,
        input_tree_images_dir_path=tree_images_dir_path,
        input_inventory_path=input_inventory_path,
        output_inventory_path=output_inventory_path,
        filetypes=filetypes,
    )
