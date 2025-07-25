import os

from treespec.utils.matching_tools import create_lists_from_shapefile, create_dictionary, create_shp_from_dict, match_and_export, match_predicted_tree_species

testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_match_and_export():
    pred_attributes_path = os.path.join(testpath, "mock/essen/run_70/matching_70/run70/run70")
    inventory_path = os.path.join(testpath, "mock/essen/run_70/matching_70/Whole-Essen/cadastre_essen")
    output_path = os.path.join(testpath, "mock/temp/matching_70/matching_70")

    match_and_export(pred_attributes_path, inventory_path, output_path)

#TODO: add missing tests