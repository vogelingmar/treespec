import shapefile
import os
from typing import Optional
from scipy.spatial import cKDTree


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



attributes_path = "/data/essen/cadastre/tree_attributes_filtered/20220905_092821_0041/20220905_092821_0041"
cadastre_path = "/data/essen/cadastre/cadastre_essen40-42/cadastre_essen"
output_path = "/data/essen/cadastre/matched_output/matched_output"
match_and_export(attributes_path, cadastre_path, output_path, True)
# print(len(create_dictionary("/data/essen/cadastre/matched_output/matched_output")))
