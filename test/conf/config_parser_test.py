import pytest
from types import SimpleNamespace
from treespec.conf import config_parser

@pytest.fixture
def mock_cfg():
    # Mock config using values from config.yaml and config.py
    return SimpleNamespace(
        train=SimpleNamespace(
            model="resnet50",
            model_weights="resnet50_default",
            dataset="folder",
            dataset_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/trees_70",
            num_classes=7,
            use_ids=True,
            epoch_count=20,
            batch_size=5,
            num_workers=27,
            learning_rate=0.001,
            loss_function="cross_entropy",
            use_augmentations=False,
            trained_model_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/models/",
            train_augmentations=[
                {"name": "RandomVerticalFlip", "p": 0.1},
                {"name": "RandomRotation", "degrees": 15},
                {"name": "RandomResizedCrop", "size": [224, 224]},
                {"name": "RandomPerspective", "distortion_scale": 0.3, "p": 0.3},
            ],
        ),
        extract=SimpleNamespace(
            model="/home/ingmar/Documents/repos/treespec/src/treespec/io/models/X-101_RGB_60k.pth",
            output_trees_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/",
            video="/data/sauen_videos/3512a1/part1/corrected_VID_20231109_182711_00_001.mp4",
            corrected=True,
            predict_video_dest_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/videos/",
            visualize=False,
            predict=False,
            mask=True,
            image_dir="/some/image/dir",
            cameras=[1, 2],
            image_filetype="jpg",
        ),
        essen_dataset=SimpleNamespace(
            attribute_path="/data/essen/inventory/matched_output_70/matched_output",
            original_color_images_path="/data/essen/data/MG4/13.09.2022/panos",
            color_images_path="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/color_70",
            color_type="jpg",
            color_output_type="png",
            original_id_images_path="/home/ingmar/Downloads/tree_filtering/output",
            segmentid_images_path="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/segmentid_70",
            seg_type="tif",
            seg_output_type="png",
            original_sem_images_path="/home/ingmar/Downloads/tree_filtering/output",
            semantic_images_path="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/semantic_70",
            sem_type="tif",
            sem_output_type="png",
            run=70,
            output_trees_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/barks_70",
            mask="bark",
            filter_id="segmentid",
            filter_semantic="semanticclass",
            crop=False,
            pictures_extracted=False,
        ),
        matching=SimpleNamespace(
            predicted_cadastre_path="/data/essen/inventory/pred_cadastre_6-82/run70/run70",
            cadastre_path="/data/essen/inventory/Whole-Essen/cadastre_essen",
            output_path="/data/essen/inventory/matched_output_70/matched_output",
            use_dbh_filter=False,
        ),
    )

def test_train_config_values(mock_cfg):
    # All possible params for train_config_values
    assert callable(config_parser.train_config_values("model", mock_cfg))
    assert hasattr(config_parser.train_config_values("model_weights", mock_cfg), "transforms")
    assert callable(config_parser.train_config_values("dataset", mock_cfg))
    assert callable(config_parser.train_config_values("loss_function", mock_cfg))
    assert hasattr(config_parser.train_config_values("train_augmentations", mock_cfg), "__call__")
    assert isinstance(config_parser.train_config_values("dataset_dir", mock_cfg), str)
    assert isinstance(config_parser.train_config_values("num_classes", mock_cfg), int)
    assert isinstance(config_parser.train_config_values("use_ids", mock_cfg), bool)
    assert isinstance(config_parser.train_config_values("epoch_count", mock_cfg), int)
    assert isinstance(config_parser.train_config_values("batch_size", mock_cfg), int)
    assert isinstance(config_parser.train_config_values("num_workers", mock_cfg), int)
    assert isinstance(config_parser.train_config_values("learning_rate", mock_cfg), float)
    assert isinstance(config_parser.train_config_values("use_augmentations", mock_cfg), bool)
    assert isinstance(config_parser.train_config_values("trained_model_dir", mock_cfg), str)

def test_image_based_extract_config_values(mock_cfg):
    # All possible params for image_based_extract_config_values
    assert isinstance(config_parser.image_based_extract_config_values("model", mock_cfg), str)
    assert isinstance(config_parser.image_based_extract_config_values("output_trees_dir", mock_cfg), str)
    assert isinstance(config_parser.image_based_extract_config_values("predict_video_dest_dir", mock_cfg), str)
    assert isinstance(config_parser.image_based_extract_config_values("visualize", mock_cfg), bool)
    assert isinstance(config_parser.image_based_extract_config_values("video", mock_cfg), str)
    assert isinstance(config_parser.image_based_extract_config_values("corrected", mock_cfg), bool)
    assert isinstance(config_parser.image_based_extract_config_values("image_dir", mock_cfg), str)
    assert isinstance(config_parser.image_based_extract_config_values("cameras", mock_cfg), list)
    assert isinstance(config_parser.image_based_extract_config_values("image_filetype", mock_cfg), str)
    assert isinstance(config_parser.image_based_extract_config_values("predict", mock_cfg), bool)
    assert isinstance(config_parser.image_based_extract_config_values("mask", mock_cfg), bool)

def test_create_essen_dataset_config_values(mock_cfg):
    # All possible params for create_essen_dataset_config_values
    assert isinstance(config_parser.create_essen_dataset_config_values("attribute_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("original_color_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("color_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("color_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("color_output_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("original_id_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("segmentid_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("seg_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("seg_output_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("original_sem_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("semantic_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("sem_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("sem_output_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("run", mock_cfg), int)
    assert isinstance(config_parser.create_essen_dataset_config_values("output_trees_dir", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("mask", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("filter_id", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("filter_semantic", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("crop", mock_cfg), bool)
    assert isinstance(config_parser.create_essen_dataset_config_values("pictures_extracted", mock_cfg), bool)

def test_matching_config_values(mock_cfg):
    # All possible params for matching_config_values
    assert isinstance(config_parser.matching_config_values("predicted_cadastre_path", mock_cfg), str)
    assert isinstance(config_parser.matching_config_values("cadastre_path", mock_cfg), str)
    assert isinstance(config_parser.matching_config_values("output_path", mock_cfg), str)
    assert isinstance(config_parser.matching_config_values("use_dbh_filter", mock_cfg), bool)
