"""Prediction function for tree species classification."""

from treespec.classification.classification_model import ClassificationModel
import pytorch_lightning as L
from treespec.dataset_creation.inventory_tools.inventory_convertion import create_dictionary_from_shapefile, create_shapefile_from_dictionary
import torch
import os
from typing import Callable
from pathlib import Path
from typing import Callable

from torch.nn.modules.loss import _Loss
from torchvision.models._api import WeightsEnum  # type: ignore


def inventurize_trees(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    input_tree_images_dir_path: Path,
    input_inventory_path: Path,
    output_inventory_path: Path,
    trained_model_path: Path,
    classification_model: ClassificationModel,
    dataset: L.LightningDataModule,
) -> None:  # pylint: disable=too-many-locals
    r"""Predicts tree species on input trees and exports predictions as an inventory shapefile.

    Args:
        input_tree_images_dir_path: Directory containing images of trees to classify.
        input_inventory_path: Path to the input inventory shapefile.
        output_inventory_path: Path to save the updated inventory shapefile with predicted species.
        trained_model_path: Path to the trained classification model weights.
        classification_model: Instance of the ClassificationModel used for prediction.
        dataset: Dataset instance containing class names for species classification.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classification_model = classification_model.to(device)
    classification_model.eval()  # Set the model to evaluation mode

    trees = create_dictionary_from_shapefile(input_inventory_path)
    class_names = dataset.classes

    for tree_name in os.listdir(input_tree_images_dir_path):
        image_path = os.path.join(input_tree_images_dir_path, tree_name)

        if os.path.isdir(image_path):
            continue

        prediction = classification_model.predict(image_path)
        predicted_class_id = prediction["category"]
        predicted_class = class_names[predicted_class_id]

        parts = os.path.splitext(tree_name)[0].split("_")

        tree_id = int(parts[0])
        image_id = str(parts[1])

        if tree_id in trees.keys():
            if "pred_species" in trees[tree_id].keys():
                trees[tree_id][f"pred_sp_{image_id}"] = predicted_class
            else:
                trees[tree_id]["pred_species"] = predicted_class
        else:
            print(f"Tree ID {tree_id} not found in the matched cadastre data. Skipping.")

    for tree in trees.values():
        number_of_votes = 0
        votes = {}
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


def predict_species(
    model: Callable,
    model_weights: WeightsEnum,
    num_classes: int,
    loss_function: _Loss,
    learning_rate: float,
    dataset: L.LightningDataModule,
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    use_ids: bool,
    tree_images_dir: Path,
    input_inventory_path: Path,
    output_inventory_path: Path,
    trained_model_path: Path,
)-> None:
    """
    Predicts tree species from images and updates the inventory shapefile with the predictions.

    This function initializes the classification model and dataset, and then calls the `match_predicted_tree_species`
    function to perform the predictions and update the inventory.

    Args:
        model: The model architecture to be used for classification.
        model_weights: Pre-trained weights for the model.
        num_classes: Number of classes for the classification task.
        loss_function: Loss function to be used during training.
        learning_rate: Learning rate for the model.
        dataset: Dataset class to be used for loading data.
        dataset_dir: Directory containing the dataset.
        batch_size: Batch size for data loading.
        num_workers: Number of workers for data loading.
        use_ids: Whether to use IDs for dataset loading.
        tree_images_dir: Directory containing images of trees to classify.
        input_inventory_path: Path to the input inventory shapefile.
        output_inventory_path: Path to save the updated inventory shapefile with predicted species.
        trained_model_path: Path to the trained classification model weights.
    """


    dataset_instance = dataset(
        dataset_dir_path=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_ids=use_ids,
    )
    default_transforms = model_weights.transforms()
    dataset_instance.prepare_data()
    dataset_instance.setup(transform=default_transforms)

    loss_function = loss_function(label_smoothing=0.1, weight=dataset_instance.loss_weights())

    classification_model = ClassificationModel.load_from_checkpoint(trained_model_path, model=model, model_weights=model_weights, num_classes=num_classes, loss_function=loss_function, learning_rate=learning_rate)

    inventurize_trees(
        classification_model=classification_model,
        dataset=dataset_instance,
        input_tree_images_dir_path=tree_images_dir,
        input_inventory_path=input_inventory_path,
        output_inventory_path=output_inventory_path,
        trained_model_path=trained_model_path,
    )
