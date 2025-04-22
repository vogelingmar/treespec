import pytest
import torch

from src.treespec.models.lumberjack import Lumberjack

@pytest.fixture
def lumberjack():
    return Lumberjack(
        model="/home/ingmar/Documents/repos/treespec/src/treespec/io/models/X-101_RGB_60k.pth",
        output_trees_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/pictures/",
        predict_video_dest_dir="/home/ingmar/Documents/repos/treespec/src/treespec/io/videos/",
        visualize=False,
    )

def test_process_video(lumberjack):
    video_path = "/data/training data/Sauen_Mapping_Dataset/cropped_20fps_Sauen.mp4"
    #lumberjack.process_video(video_path, True)
    pass