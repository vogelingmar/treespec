"""Inventory conversion tools for dataset creation and prediction matching."""

import shapefile  # type: ignore
from pathlib import Path
import os
from typing import Optional


def create_lists_from_shapefile(shapefile_path: Path, prefix_to_be_applied: Optional[str]) -> tuple[list, list]:
    r"""Create lists of points and records from a shapefile.

    Args:
        shapefile_path: Path to the shapefile (without .shp extension).
        prefix_to_be_applied: Prefix to put in front of the keys in the records.

    Returns:
        Tuple containing:
            points: List of points from the shapefile.
            records: List of records from the shapefile, with keys prefixed if specified.
    """
    points = shapefile.Reader(shapefile_path)
    try:
        points_shape_records = points.shapeRecords()
    except UnboundLocalError:
        # Happens with 3D MultipointZ without M-values
        print(f"⚠️ Warning: Missing M-values in shapefile '{shapefile_path}', using fallback reader.")
        points_shape_records = list(points.iterShapeRecords())
    points = []
    records = []
    for shaperec in points_shape_records:
        for point in shaperec.shape.points:
            points.append(point)
            if prefix_to_be_applied is not None:
                record = {f"{prefix_to_be_applied}_{k}": v for k, v in shaperec.record.as_dict().items()}
            else:
                record = shaperec.record.as_dict()
            records.append(record)

    return points, records


def create_dictionary_from_shapefile(shapefile_path: Path) -> dict:
    r"""Create a dictionary from a shapefile where keys are predicted tree IDs.

    Args:
        shapefile_path: Path to the shapefile.

    Returns:
        attributes: Dictionary where keys are predicted tree IDs and values are records from the shapefile.
    """
    points, records = create_lists_from_shapefile(shapefile_path, None)
    attributes = {}
    for i, point in enumerate(points):  # pylint: disable=unused-variable
        pred_tree_id = records[i].get("pred_id")
        coordinate_dict = {"X": point[0], "Y": point[1]}
        if pred_tree_id is not None:
            attributes[pred_tree_id] = records[i] | coordinate_dict

    return attributes


def create_shapefile_from_dictionary(dictionary: dict, output_shapefile_path: Path) -> None:
    r"""Create a shapefile from a dictionary where keys are predicted tree IDs and save it to the output_shapefile_path.

    Args:
        dictionary: Dictionary where keys are predicted tree IDs and values are records.
        output_shapefile_path: Path to save the shapefile (without .shp extension).

    Raises:
        ValueError: If coordinate keys are not found in the records.
        ValueError: If the input dictionary is empty.
    """
    os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)
    if not dictionary:
        raise ValueError("Input dictionary is empty.")

    # Infer fields from the first record
    first_record = next(iter(dictionary.values()))
    if "X" not in first_record or "Y" not in first_record:
        raise ValueError("Coordinate keys 'X' and 'Y' not found in the records.")

    # Prepare fields (exclude coordinates)
    fields = [(k, "C", 50, 0) for k in first_record.keys() if k not in ("X", "Y")]

    w = shapefile.Writer(output_shapefile_path, shapeType=shapefile.POINT)
    for field in fields:
        w.field(*field)

    for record in dictionary.values():
        x, y = record["X"], record["Y"]
        w.point(x, y)
        rec = [record.get(k, None) for k in first_record.keys() if k not in ("X", "Y")]
        w.record(*rec)

    w.close()
    print(f"Exported {len(dictionary)} points to {output_shapefile_path}.shp")
