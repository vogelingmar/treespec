"""Tests for the inventory_convertion module."""

import os
import shutil

from treespec.dataset_creation.inventory_tools.inventory_convertion import (
    create_lists_from_shapefile,
    create_dictionary_from_shapefile,
    create_shapefile_from_dictionary,
)

dataset_creation_mock_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")


def test_create_lists_from_shapefile():
    """Tests the create_lists_from_shapefile function by checking
    if the output lists contain points and records with the expected prefix."""
    matched_tree_inventory_output_path = os.path.join(
        dataset_creation_mock_path, "inventories", "inventory_matched", "matched_output"
    )
    prefix = "test"

    points, records = create_lists_from_shapefile(matched_tree_inventory_output_path, prefix)

    assert len(points) > 0
    assert len(records) > 0
    for record in records:
        for key in record.keys():
            assert key.startswith(prefix + "_")


def test_create_dictionary():
    """Tests the create_dictionary function by checking if the output dictionary contains
    attributes with integer keys and 'pred_id'."""
    matched_tree_inventory_output_path = os.path.join(
        dataset_creation_mock_path, "inventories", "inventory_matched", "matched_output"
    )

    attributes = create_dictionary_from_shapefile(matched_tree_inventory_output_path)
    assert len(attributes) > 0
    for key in attributes.items():
        assert isinstance(key, int)
        assert "pred_id" in attributes[key]


def test_create_shp_from_dict():
    """Tests the create_shp_from_dict function by checking if the output
    shapefile and its associated files are created successfully."""
    dictionary = {
        "14": {
            "ANGELEDAT": "",
            "AREA": "1865061.0818356618",
            "BAUMART": "SiAh",
            "BAUMID": "22385",
            "BAUMKENNZ": "",
            "BAUMNR": "3",
            "CNTX": "32360449.222882047",
            "CNTY": "5695932.225481497",
            "DURCHM": "104.0",
            "DURCHM2": "0.0",
            "DURCHM3": "0.0",
            "GEAENDDAT": "",
            "GEFAELLT": "",
            "GRUNDNR": "1741",
            "HEIGHT": "2494.2518188832328",
            "HINDERNIS": "",
            "HOCH": "5697158.6733",
            "HOEHE": "22.0",
            "JAHR0": "1922",
            "KINTERVALL": "",
            "KRONBREITE": "14.0",
            "KTERMIN": "",
            "MAXX": "32360823.094731662",
            "MAXY": "5697179.351390938",
            "MINX": "32360075.35103244",
            "MINY": "5694685.099572055",
            "ORTSBESCH": "",
            "PERIM": "6483.991036204621",
            "RECHTS": "32360373.0027",
            "RISIKO": "",
            "WE": "1000",
            "WIDTH": "747.7436992190779",
            "X": 1.0,
            "Y": 1.0,
            "layer": "cadastre_essen",
            "path": "/data/essen/cadastre/cadastre_essen40-42/cadastre_",
            "pred_cbh": "2.612",
            "pred_cv": "52.339",
            "pred_cw": "4.796",
            "pred_d_1_m": "0.161",
            "pred_d_2_m": "0.326",
            "pred_d_3_m": "0.137",
            "pred_dbh": "0.168",
            "pred_dir_x": "-0.004",
            "pred_dir_y": "-0.019",
            "pred_dir_z": "1.0",
            "pred_heigh": "9.025",
            "pred_id": "14",
            "pred_pos_1": "5697159.648",
            "pred_pos_2": "5697159.623",
            "pred_pos_3": "5697159.64",
            "pred_pos_d": "5697159.636",
            "pred_ubh": "2.206",
        },
        "18": {
            "ANGELEDAT": "",
            "AREA": "1865061.0818356618",
            "BAUMART": "MeB",
            "BAUMID": "40931",
            "BAUMKENNZ": "",
            "BAUMNR": "19",
            "CNTX": "32360449.222882047",
            "CNTY": "5695932.225481497",
            "DURCHM": "22.0",
            "DURCHM2": "0.0",
            "DURCHM3": "0.0",
            "GEAENDDAT": "",
            "GEFAELLT": "",
            "GRUNDNR": "2686",
            "HEIGHT": "2494.2518188832328",
            "HINDERNIS": "",
            "HOCH": "5697177.7434",
            "HOEHE": "6.0",
            "JAHR0": "1993",
            "KINTERVALL": "",
            "KRONBREITE": "4.0",
            "KTERMIN": "",
            "MAXX": "32360823.094731662",
            "MAXY": "5697179.351390938",
            "MINX": "32360075.35103244",
            "MINY": "5694685.099572055",
            "ORTSBESCH": "",
            "PERIM": "6483.991036204621",
            "RECHTS": "32360371.8847",
            "RISIKO": "",
            "WE": "1000",
            "WIDTH": "747.7436992190779",
            "X": 2.0,
            "Y": 2.0,
            "layer": "cadastre_essen",
            "path": "/data/essen/cadastre/cadastre_essen40-42/cadastre_",
            "pred_cbh": "5.843",
            "pred_cv": "12.798",
            "pred_cw": "5.647",
            "pred_d_1_m": "0.089",
            "pred_d_2_m": "",
            "pred_d_3_m": "",
            "pred_dbh": "0.081",
            "pred_dir_x": "0.515",
            "pred_dir_y": "-0.163",
            "pred_dir_z": "0.841",
            "pred_heigh": "5.843",
            "pred_id": "18",
            "pred_pos_1": "5697174.274",
            "pred_pos_2": "",
            "pred_pos_3": "",
            "pred_pos_d": "5697174.295",
            "pred_ubh": "5.843",
        },
    }

    output_path = os.path.join(dataset_creation_mock_path, "temp/shapefile_output/example")

    shutil.rmtree(os.path.dirname(output_path), ignore_errors=True)

    create_shapefile_from_dictionary(dictionary, output_path)

    assert os.path.exists(output_path + ".shp")
    assert os.path.exists(output_path + ".dbf")
    assert os.path.exists(output_path + ".shx")

    attributes = create_dictionary_from_shapefile(output_path)
    assert attributes == dictionary

    shutil.rmtree(os.path.dirname(output_path), ignore_errors=True)
