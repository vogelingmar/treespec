"""Tests for the match_inventories function."""

import os
import shutil

from treespec.dataset_creation.functions.match_inventories import match
from treespec.dataset_creation.inventory_tools.inventory_convertion import create_dictionary_from_shapefile


def test_match():
    dataset_creation_mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")
    predicted_tree_inventory_path = os.path.join(
        dataset_creation_mock_path, "inventories", "inventory_predicted", "run70"
    )
    groundtruth_tree_inventory_path = os.path.join(
        dataset_creation_mock_path, "inventories", "inventory_groundtruth", "cadastre_essen"
    )
    matched_tree_inventory_output_path = os.path.join(
        dataset_creation_mock_path, "temp", "matched_inventory", "matched_inventory"
    )
    matched_tree_inventory_output_dir = os.path.dirname(matched_tree_inventory_output_path)
    use_dbh_matching_filter = True

    shutil.rmtree(matched_tree_inventory_output_dir, ignore_errors=True)

    match(
        predicted_inventory_path=predicted_tree_inventory_path,
        groundtruth_inventory_path=groundtruth_tree_inventory_path,
        output_inventory_path=matched_tree_inventory_output_path,
        use_dbh_filter=use_dbh_matching_filter,
    )
    assert os.path.exists(matched_tree_inventory_output_path + ".shp")
    assert os.path.exists(matched_tree_inventory_output_path + ".dbf")
    assert os.path.exists(matched_tree_inventory_output_path + ".shx")

    matched_inventory = create_dictionary_from_shapefile(matched_tree_inventory_output_path)
    assert len(matched_inventory) > 0
    for tree in matched_inventory.values():
        assert "pred_id" in tree.keys()
        assert "BAUMART" in tree.keys()

    shutil.rmtree(matched_tree_inventory_output_dir, ignore_errors=True)
