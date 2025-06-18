import shapefile
import os
from typing import Optional
from scipy.spatial import cKDTree

import torch


from treespec.models.classification_model import ClassificationModel


from treespec.conf.config import TreespecConfig
from treespec.conf.config_parser import train_config_values

def create_lists_from_shapefile(path: str, prefix: Optional[str]):
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
    points, records = create_lists_from_shapefile(path, None)
    attributes = {}
    for i, point in enumerate(points):
        pred_tree_id = records[i].get("pred_id")
        if pred_tree_id is not None:
            attributes[pred_tree_id] = records[i]

    return attributes


def create_shp_from_dict(dictionary: dict, output_path: str):
    #not tested yet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get the first record to infer fields and coordinate keys
    first_record = next(iter(dictionary.values()))
    # Try common coordinate keys
    if "X" in first_record and "Y" in first_record:
        x_key, y_key = "X", "Y"
    elif "x" in first_record and "y" in first_record:
        x_key, y_key = "x", "y"
    elif "POINT_X" in first_record and "POINT_Y" in first_record:
        x_key, y_key = "POINT_X", "POINT_Y"
    else:
        raise ValueError("Could not find coordinate keys in the dictionary records.")

    # Prepare fields (skip coordinate keys)
    fields = [(k, "C", 50, 0) for k in first_record.keys() if k not in [x_key, y_key]]

    w = shapefile.Writer(output_path, shapeType=shapefile.POINT)
    for name, ftype, size, dec in fields:
        w.field(name, ftype, size, dec)

    for rec in dictionary.values():
        x, y = rec[x_key], rec[y_key]
        w.point(x, y)
        # Write attributes in the same order as fields
        w.record(*[rec.get(k) for k, *_ in fields])

    w.close()
    print(f"Created shapefile at {output_path}.shp")


def match_and_export(attributes_path: str, cadastre_path: str, output_path: str, use_dbh_filter: bool = True):

    os.makedirs(output_path, exist_ok=True)

    cadastre = shapefile.Reader(cadastre_path)
    attributes = shapefile.Reader(attributes_path)

    attribute_points, attribute_records = create_lists_from_shapefile(attributes_path, "pred")
    cadastre_points, cadastre_records = create_lists_from_shapefile(cadastre_path, None)

    # Build KDTree for cadastre points
    cadastre_tree = cKDTree(cadastre_points)
    attribute_tree = cKDTree(attribute_points)
    # Find nearest neighbor in cadastre for each attribute point
    cad_distances, cad_indices = cadastre_tree.query(attribute_points)
    # Find nearest neighbor in attributes for each cadastre point
    att_distances, att_indices = attribute_tree.query(cadastre_points)
    # Prepare merged data: (cadastre_point, merged_record)
    merged = []
    for i, (attr_pt, cad_idx, cad_dist) in enumerate(zip(attribute_points, cad_indices, cad_distances)):
        if cad_dist <= 5.0 and att_indices[cad_idx] == i:
            combined = {**cadastre_records[cad_idx], **attribute_records[i]}
            if not use_dbh_filter:
                merged.append((cadastre_points[cad_idx], combined))
            elif combined['pred_dbh'] is not None and (combined['DURCHM'] - combined['pred_dbh'] * 100) < 10:
                # Add the matched point to the merged list
                merged.append((cadastre_points[cad_idx], combined))
    # Prepare fields for output shapefile
    w = shapefile.Writer(output_path, shapeType=shapefile.POINT)
    # Add cadastre fields
    for field in cadastre.fields[1:]:  # skip DeletionFlag
        w.field(*field)
    # Add attribute fields with pred_ prefix
    for field in attributes.fields[1:]:
        w.field(f"pred_{field[0]}", field[1], field[2], field[3])
    # Write merged points and records
    for pt, combined in merged:
        w.point(*pt)
        record = [combined.get(f[0], None) for f in cadastre.fields[1:]] + [
            combined.get(f"pred_{f[0]}", None) for f in attributes.fields[1:]
        ]
        w.record(*record)
    w.close()
    print(f"Exported matched points to {output_path}.shp")


def match_predicted_tree_species(tree_images_dir, matched_cadastre_path):
    #not tested yet - need data
    
    classification_model = ClassificationModel(
        model=train_config_values("model"),
        model_weights=train_config_values("model_weights"),
        num_classes=train_config_values("num_classes"),
        loss_function=train_config_values("loss_function")(),
        learning_rate=train_config_values("learning_rate"),
    )

    dataset = train_config_values("dataset")(
            data_dir=train_config_values("dataset_dir"),
            batch_size=train_config_values("batch_size"),
            num_workers=train_config_values("num_workers"),
        )

    trained_model_path = train_config_values("trained_model_dir") + train_config_values("model") + "_finetuned" + ".pth"
    classification_model.model.load_state_dict(torch.load(trained_model_path))
    classification_model.eval()  # Set the model to evaluation mode

    trees = create_dictionary(matched_cadastre_path)
    class_names = dataset.classes

    for tree_name in os.listdir(tree_images_dir):
        image_path = os.path.join(tree_images_dir, tree_name)

        if os.path.isdir(image_path):
            continue

        prediction = classification_model.predict(image_path)
        predicted_class_id = prediction["category"]
        predicted_class = class_names[predicted_class_id]

        parts = os.path.splitext(tree_name)[0].split('_')

        tree_id = parts[0]

        if tree_id in trees:
            if "pred_species" in trees[tree_id].keys():
                trees[tree_id][f"pred_species_{parts[1]}"] = predicted_class
            else:
                trees[tree_id]["pred_species"] = predicted_class
        else:
            raise ValueError(f"Tree ID {tree_id} not found in the matched cadastre data.")
        
    for tree in trees:
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
            if votes[species] > number_of_votes/2:
                trees[tree]["pred_species"] = species
            else: 
                trees[tree]["pred_species"] = "unknown"
    create_shp_from_dict(trees, matched_cadastre_path + "_w_pred_species")


attributes_path = "/data/essen/cadastre/tree_attributes_filtered/20220905_092821_0041/20220905_092821_0041"
cadastre_path = "/data/essen/cadastre/cadastre_essen40-42/cadastre_essen"
output_path = "/data/essen/cadastre/matched_output/matched_output"
match_and_export(attributes_path, cadastre_path, output_path, True)
# print(len(create_dictionary("/data/essen/cadastre/matched_output/matched_output")))
