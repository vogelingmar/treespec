import os
import shutil

from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from treespec.utils.matching_tools import create_lists_from_shapefile, create_dictionary, create_shp_from_dict, match_and_export, match_predicted_tree_species
from treespec.datasets.image_dataset import ImageDataset
from treespec.models.classification_model import ClassificationModel

testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_match_and_export():
    """Tests the match_and_export function by checking the output files if there are attributes from both inventories."""
    pred_attributes_path = os.path.join(testpath, "mock/essen_mock/run_70/matching_70/run70/run70")
    inventory_path = os.path.join(testpath, "mock/essen_mock/run_70/matching_70/Whole-Essen/cadastre_essen")
    output_file = os.path.join(testpath, "mock/temp/matching_70/matching_70")
    output_dir = os.path.dirname(output_file)

    shutil.rmtree(output_dir, ignore_errors=True)

    match_and_export(pred_attributes_path, inventory_path, output_file)
    assert os.path.exists(output_file + ".shp")
    assert os.path.exists(output_file + ".dbf")
    assert os.path.exists(output_file + ".shx")

    matched_inventory = create_dictionary(output_file)
    assert len(matched_inventory) > 0
    for tree in matched_inventory.values():
        assert "pred_id" in tree.keys()
        assert "BAUMART" in tree.keys()

    shutil.rmtree(output_dir, ignore_errors=True)

def test_create_lists_from_shapefile():
    """Tests the create_lists_from_shapefile function by checking if the output lists contain points and records with the expected prefix."""
    path = os.path.join(testpath, "mock/essen_mock/run_70/inventory_70/matched_output")
    prefix = "test"
    
    points, records = create_lists_from_shapefile(path, prefix)

    assert len(points) > 0
    assert len(records) > 0
    for record in records:
        for key in record.keys():
            assert key.startswith(prefix + "_")

def test_create_dictionary():
    """Tests the create_dictionary function by checking if the output dictionary contains attributes with integer keys and 'pred_id'."""
    path = os.path.join(testpath, "mock/essen_mock/run_70/inventory_70/matched_output")
    
    attributes = create_dictionary(path)
    assert len(attributes) > 0
    for key in attributes.keys():
        assert isinstance(key, int)
        assert "pred_id" in attributes[key]

def test_create_shp_from_dict():
    """Tests the create_shp_from_dict function by checking if the output shapefile and its associated files are created successfully."""
    dictionary = {
        '14': {'ANGELEDAT': '',
          'AREA': '1865061.0818356618',
          'BAUMART': 'SiAh',
          'BAUMID': '22385',
          'BAUMKENNZ': '',
          'BAUMNR': '3',
          'CNTX': '32360449.222882047',
          'CNTY': '5695932.225481497',
          'DURCHM': '104.0',
          'DURCHM2': '0.0',
          'DURCHM3': '0.0',
          'GEAENDDAT': '',
          'GEFAELLT': '',
          'GRUNDNR': '1741',
          'HEIGHT': '2494.2518188832328',
          'HINDERNIS': '',
          'HOCH': '5697158.6733',
          'HOEHE': '22.0',
          'JAHR0': '1922',
          'KINTERVALL': '',
          'KRONBREITE': '14.0',
          'KTERMIN': '',
          'MAXX': '32360823.094731662',
          'MAXY': '5697179.351390938',
          'MINX': '32360075.35103244',
          'MINY': '5694685.099572055',
          'ORTSBESCH': '',
          'PERIM': '6483.991036204621',
          'RECHTS': '32360373.0027',
          'RISIKO': '',
          'WE': '1000',
          'WIDTH': '747.7436992190779',
          'X': 1.0,
          'Y': 1.0,
          'layer': 'cadastre_essen',
          'path': '/data/essen/cadastre/cadastre_essen40-42/cadastre_',
          'pred_cbh': '2.612',
          'pred_cv': '52.339',
          'pred_cw': '4.796',
          'pred_d_1_m': '0.161',
          'pred_d_2_m': '0.326',
          'pred_d_3_m': '0.137',
          'pred_dbh': '0.168',
          'pred_dir_x': '-0.004',
          'pred_dir_y': '-0.019',
          'pred_dir_z': '1.0',
          'pred_heigh': '9.025',
          'pred_id': '14',
          'pred_pos_1': '5697159.648',
          'pred_pos_2': '5697159.623',
          'pred_pos_3': '5697159.64',
          'pred_pos_d': '5697159.636',
          'pred_ubh': '2.206'},
   '18': {'ANGELEDAT': '',
          'AREA': '1865061.0818356618',
          'BAUMART': 'MeB',
          'BAUMID': '40931',
          'BAUMKENNZ': '',
          'BAUMNR': '19',
          'CNTX': '32360449.222882047',
          'CNTY': '5695932.225481497',
          'DURCHM': '22.0',
          'DURCHM2': '0.0',
          'DURCHM3': '0.0',
          'GEAENDDAT': '',
          'GEFAELLT': '',
          'GRUNDNR': '2686',
          'HEIGHT': '2494.2518188832328',
          'HINDERNIS': '',
          'HOCH': '5697177.7434',
          'HOEHE': '6.0',
          'JAHR0': '1993',
          'KINTERVALL': '',
          'KRONBREITE': '4.0',
          'KTERMIN': '',
          'MAXX': '32360823.094731662',
          'MAXY': '5697179.351390938',
          'MINX': '32360075.35103244',
          'MINY': '5694685.099572055',
          'ORTSBESCH': '',
          'PERIM': '6483.991036204621',
          'RECHTS': '32360371.8847',
          'RISIKO': '',
          'WE': '1000',
          'WIDTH': '747.7436992190779',
          'X': 2.0,
          'Y': 2.0,
          'layer': 'cadastre_essen',
          'path': '/data/essen/cadastre/cadastre_essen40-42/cadastre_',
          'pred_cbh': '5.843',
          'pred_cv': '12.798',
          'pred_cw': '5.647',
          'pred_d_1_m': '0.089',
          'pred_d_2_m': '',
          'pred_d_3_m': '',
          'pred_dbh': '0.081',
          'pred_dir_x': '0.515',
          'pred_dir_y': '-0.163',
          'pred_dir_z': '0.841',
          'pred_heigh': '5.843',
          'pred_id': '18',
          'pred_pos_1': '5697174.274',
          'pred_pos_2': '',
          'pred_pos_3': '',
          'pred_pos_d': '5697174.295',
          'pred_ubh': '5.843'}}

    output_path = os.path.join(testpath, "mock/temp/shapefile_output/example")
    
    shutil.rmtree(os.path.dirname(output_path), ignore_errors=True)

    create_shp_from_dict(dictionary, output_path)
    
    assert os.path.exists(output_path + ".shp")
    assert os.path.exists(output_path + ".dbf")
    assert os.path.exists(output_path + ".shx")

    attributes = create_dictionary(output_path)
    assert attributes == dictionary
    
    shutil.rmtree(os.path.dirname(output_path), ignore_errors=True)

def test_match_predicted_tree_species():
    """Tests the match_predicted_tree_species function."""
    tree_images_dir = os.path.join(testpath, "mock/essen_mock/run_70/trees_70")
    input_inventory_path = os.path.join(testpath, "mock/essen_mock/run_70/inventory_70/matched_output")
    output_inventory_path = os.path.join(testpath, "mock/temp/predicted_tree_species_inventory/pred")
    trained_model_path = os.path.join(testpath, "mock/essen_mock/trained_models/resnet50trees_70_finetuned.pth")

    dataset = ImageDataset(data_dir=os.path.join(testpath, "mock/essen_mock/run_70/dataset_70"),
                           batch_size=5, num_workers=2, use_ids=True)
    classification_model = ClassificationModel(
        model=resnet50,
        model_weights=ResNet50_Weights.DEFAULT,
        num_classes=3,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=0.001,
    )


    shutil.rmtree(os.path.dirname(output_inventory_path), ignore_errors=True)

    match_predicted_tree_species(tree_images_dir=tree_images_dir, input_inventory_path=input_inventory_path, output_inventory_path=output_inventory_path, trained_model_path=trained_model_path, classification_model=classification_model, dataset=dataset)
    
    assert os.path.exists(output_inventory_path + ".shp")
    assert os.path.exists(output_inventory_path + ".dbf")
    assert os.path.exists(output_inventory_path + ".shx")

    inventory = create_dictionary(output_inventory_path)

    assert len(inventory) > 0

    for tree in inventory.values():
        assert "pred_speci" in tree.keys()
        assert tree["pred_speci"] is not None or tree["pred_speci"] != ""

    shutil.rmtree(os.path.dirname(output_inventory_path), ignore_errors=True)