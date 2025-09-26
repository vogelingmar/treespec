from treespec.dataset_creation.inventory_tools.inventory_convertion import create_lists_from_shapefile, create_shapefile_from_dictionary
from scipy.spatial import cKDTree  # type: ignore
from pathlib import Path

def match(  # pylint: disable=too-many-locals
    predicted_inventory_path: Path,
    groundtruth_inventory_path: Path,
    output_inventory_path: Path,
    use_dbh_filter: bool = True,
)-> None:
    r"""Match predicted cadastre with inventory points and export to a new shapefile at output_path.

    Args:
        predicted_inventory_path: Path to the predicted inventory shapefile (without .shp extension).
        groundtruth_inventory_path: Path to the inventory shapefile (without .shp extension).
        output_inventory_path: Path to save the matched shapefile (without .shp extension).
        use_dbh_filter: If True, only match points where the predicted DBH is within 10 cm of the groundtruth DBH.
    """

    attribute_points, attribute_records = create_lists_from_shapefile(predicted_inventory_path, "pred")
    cadastre_points, cadastre_records = create_lists_from_shapefile(groundtruth_inventory_path, None)

    cadastre_tree = cKDTree(cadastre_points)
    attribute_tree = cKDTree(attribute_points)
    cad_distances, cad_indices = cadastre_tree.query(attribute_points)
    _, att_indices = attribute_tree.query(cadastre_points)

    merged_dict = {}
    for i, (_, cad_idx, cad_dist) in enumerate(zip(attribute_points, cad_indices, cad_distances)):
        if cad_dist <= 5.0 and att_indices[cad_idx] == i:
            combined = {**cadastre_records[cad_idx], **attribute_records[i]}
            x, y = cadastre_points[cad_idx]
            combined["X"] = x
            combined["Y"] = y
            if not use_dbh_filter:
                merged_dict[i] = combined
            elif (
                combined.get("pred_dbh") is not None
                and (float(combined["DURCHM"]) - float(combined["pred_dbh"]) * 100) < 10
            ):
                merged_dict[i] = combined

    create_shapefile_from_dictionary(merged_dict, output_inventory_path)
    print(f"Exported matched points to {output_inventory_path}.shp")
