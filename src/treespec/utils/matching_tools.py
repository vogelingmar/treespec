"""Matching tools for cadastre shapefile and predicted_cadastre shapefile."""

import shapefile  # type: ignore
import os
from typing import Optional
from scipy.spatial import cKDTree  # type: ignore
import torch

from treespec.models.classification_model import ClassificationModel

from treespec.conf.config_parser import train_config_values

def create_lists_from_shapefile(path: str, prefix: Optional[str]):
    """Create lists of points and records from a shapefile.

    Args:
        path: Path to the shapefile.
        prefix: Prefix to put in front of the keys in the records.

    Returns:
        points: List of points from the shapefile.
        records: List of records from the shapefile, with keys prefixed if specified.
    """
    points = shapefile.Reader(path)
    points_shape_records = points.shapeRecords()
    points = []
    records = []
    for shaperec in points_shape_records:
        for point in shaperec.shape.points:
            points.append(point)
            if prefix is not None:
                record = {f"{prefix}_{k}": v for k, v in shaperec.record.as_dict().items()}
            else:
                record = shaperec.record.as_dict()
            records.append(record)

    return points, records


def create_dictionary(path: str):
    """Create a dictionary from a shapefile where keys are predicted tree IDs.

    Args:
        path: Path to the shapefile.

    Returns:
        attributes: Dictionary where keys are predicted tree IDs and values are records from the shapefile.
    """
    points, records = create_lists_from_shapefile(path, None)
    attributes = {}
    for i, point in enumerate(points):  # pylint: disable=unused-variable
        pred_tree_id = records[i].get("pred_id")
        coordinate_dict = {"X": point[0], "Y": point[1]}
        if pred_tree_id is not None:
            attributes[pred_tree_id] = records[i] | coordinate_dict

    return attributes


def create_shp_from_dict(dictionary: dict, output_path: str):
    """Create a shapefile from a dictionary where keys are predicted tree IDs and save it to the output_path.

    Args:
        dictionary: Dictionary where keys are predicted tree IDs and values are records.
        output_path: Path to save the shapefile (without extension).

    Raises:
        ValueError: If coordinate keys are not found in the records.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not dictionary:
        raise ValueError("Input dictionary is empty.")

    # Infer fields from the first record
    first_record = next(iter(dictionary.values()))
    if "X" not in first_record or "Y" not in first_record:
        raise ValueError("Coordinate keys 'X' and 'Y' not found in the records.")

    # Prepare fields (exclude coordinates)
    fields = [(k, "C", 50, 0) for k in first_record.keys() if k not in ("X", "Y")]

    w = shapefile.Writer(output_path, shapeType=shapefile.POINT)
    for field in fields:
        w.field(*field)

    for record in dictionary.values():
        x, y = record["X"], record["Y"]
        w.point(x, y)
        rec = [record.get(k, None) for k in first_record.keys() if k not in ("X", "Y")]
        w.record(*rec)

    w.close()
    print(f"Exported {len(dictionary)} points to {output_path}.shp")


def match_and_export(
    attributes_path: str,
    cadastre_path: str,
    output_path: str,
    use_dbh_filter: bool = True,
):
    """Match predicted cadastre with cadastre points and export to a new shapefile at output_path."""
    cadastre = shapefile.Reader(cadastre_path)
    attributes = shapefile.Reader(attributes_path)

    attribute_points, attribute_records = create_lists_from_shapefile(attributes_path, "pred")
    cadastre_points, cadastre_records = create_lists_from_shapefile(cadastre_path, None)

    cadastre_tree = cKDTree(cadastre_points)
    attribute_tree = cKDTree(attribute_points)
    cad_distances, cad_indices = cadastre_tree.query(attribute_points)
    att_distances, att_indices = attribute_tree.query(cadastre_points)

    merged_dict = {}
    for i, (attr_pt, cad_idx, cad_dist) in enumerate(
        zip(attribute_points, cad_indices, cad_distances)
    ):
        if cad_dist <= 5.0 and att_indices[cad_idx] == i:
            combined = {**cadastre_records[cad_idx], **attribute_records[i]}
            x, y = cadastre_points[cad_idx]
            combined["X"] = x
            combined["Y"] = y
            if not use_dbh_filter:
                merged_dict[i] = combined
            elif combined.get("pred_dbh") is not None and (float(combined["DURCHM"]) - float(combined["pred_dbh"]) * 100) < 10:
                merged_dict[i] = combined

    create_shp_from_dict(merged_dict, output_path)
    print(f"Exported matched points to {output_path}.shp")


def match_predicted_tree_species(tree_images_dir, input_inventory_path, output_inventory_path, cfg):  # pylint: disable=too-many-locals
    """Match predicted tree species from images to the matched cadastre shapefile and writes it to the input path.

    Args:
        tree_images_dir: Directory containing images of trees to classify.
        matched_cadastre_path: Path to the matched cadastre shapefile.
        cfg: Configuration object containing model and dataset parameters.

    Raises:
        ValueError: If a tree ID from the images is not found in the matched cadastre data.
    """
    # not tested yet - need data

    classification_model = ClassificationModel(
        model=train_config_values("model", cfg),
        model_weights=train_config_values("model_weights", cfg),
        num_classes=train_config_values("num_classes", cfg),
        loss_function=train_config_values("loss_function", cfg)(),
        learning_rate=train_config_values("learning_rate", cfg),
    )

    dataset = train_config_values("dataset", cfg)(
        data_dir=train_config_values("dataset_dir", cfg),
        batch_size=train_config_values("batch_size", cfg),
        num_workers=train_config_values("num_workers", cfg),
    )

    trained_model_path = (
        train_config_values("trained_model_dir", cfg) + train_config_values("model", cfg) + "_finetuned" + ".pth"
    )
    classification_model.model.load_state_dict(torch.load(trained_model_path))
    classification_model.eval()  # Set the model to evaluation mode

    trees = create_dictionary(input_inventory_path)
    class_names = dataset.classes

    for tree_name in os.listdir(tree_images_dir):
        image_path = os.path.join(tree_images_dir, tree_name)

        if os.path.isdir(image_path):
            continue

        prediction = classification_model.predict(image_path)
        predicted_class_id = prediction["category"]
        predicted_class = class_names[predicted_class_id]

        parts = os.path.splitext(tree_name)[0].split("_")

        tree_id = parts[0]

        if tree_id in trees:
            if "pred_species" in trees[tree_id].keys():
                trees[tree_id][f"pred_species_{parts[1]}"] = predicted_class
            else:
                trees[tree_id]["pred_species"] = predicted_class
        else:
            raise ValueError(f"Tree ID {tree_id} not found in the matched cadastre data.")

    for tree in trees.items():
        number_of_votes = 0
        votes = {}
        attributes = list(trees[tree].keys())
        for attribute in attributes:
            if attribute.startswith("pred_species"):
                number_of_votes += 1
                species = trees[tree][attribute]
                if species not in votes.keys():
                    votes[species] = 1
                elif species in votes.keys():
                    votes[species] += 1
        for species in votes.keys():
            if votes[species] > number_of_votes / 2:
                trees[tree]["pred_species"] = species
            else:
                trees[tree]["pred_species"] = "unknown"
    create_shp_from_dict(trees, input_inventory_path + "_w_pred_species")

