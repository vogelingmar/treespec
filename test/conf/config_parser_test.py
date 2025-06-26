import pytest
from types import SimpleNamespace
from treespec.conf import config_parser

@pytest.fixture
def mock_cfg():
    # Mock config using values from config.yaml
    return SimpleNamespace(
        train=SimpleNamespace(
            model="resnet152",
            model_weights="resnet152_default",
            dataset="folder",
            dataset_dir="/home/ingmar/Documents/repos/treespec/src/treespec/datasets/sauen/sauen_big_clean_nv6",
            num_classes=9,
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
            attribute_path="/data/essen/cadastre/matched_output/matched_output",
            original_color_images_path="/data/essen/MG4/2022-09-12_pano",
            color_images_path="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/color_4",
            color_type="jpg",
            original_seg_images_path="/data/essen/MG4/2022-09-12_2_seg",
            segmentid_images_path="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/segmentid_4",
            seg_type="tif",
            seg_output_type="png",
            run=4,
            output_trees_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/trees_4",
            mask=True,
            filter="segmentid",
            crop=False,
        ),
        matching=SimpleNamespace(
            predicted_cadastre_path="/data/essen/cadastre/tree_attributes_filtered/20220905_092821_0041/20220905_092821_0041",
            cadastre_path="/data/essen/cadastre/cadastre_essen40-42/cadastre_essen",
            output_path="/data/essen/cadastre/matched_output/matched_output",
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
    assert isinstance(config_parser.create_essen_dataset_config_values("original_color_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("color_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("color_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("original_seg_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("segmentid_images_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("seg_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("seg_output_type", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("run", mock_cfg), int)
    assert isinstance(config_parser.create_essen_dataset_config_values("output_trees_dir", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("attribute_path", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("mask", mock_cfg), bool)
    assert isinstance(config_parser.create_essen_dataset_config_values("filter", mock_cfg), str)
    assert isinstance(config_parser.create_essen_dataset_config_values("crop", mock_cfg), bool)

def test_matching_config_values(mock_cfg):
    # All possible params for matching_config_values
    assert isinstance(config_parser.matching_config_values("predicted_cadastre_path", mock_cfg), str)
    assert isinstance(config_parser.matching_config_values("cadastre_path", mock_cfg), str)
    assert isinstance(config_parser.matching_config_values("output_path", mock_cfg), str)
