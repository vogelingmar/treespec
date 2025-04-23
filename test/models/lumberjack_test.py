import os
import pytest
import torch

from src.treespec.models.lumberjack import Lumberjack

@pytest.fixture
def lumberjack():
    if torch.cuda.is_available():
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return Lumberjack(
            model = os.path.join(base_path, "src/treespec/io/models/X-101_RGB_60k.pth"),
            output_trees_dir = os.path.join(base_path, "test/mock/temp/pictures/"),
            predict_video_dest_dir = os.path.join(base_path, "test/mock/temp/videos/"),
            visualize=True,
        )

def test_process_video(lumberjack):
    if not torch.cuda.is_available():
        pass
    
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_path, "mock/10sec_example.mp4")

        output_dir = lumberjack.output_trees_dir
        predict_dir = lumberjack.predict_video_dest_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))

        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
        else:
            for file in os.listdir(predict_dir):
                os.remove(os.path.join(predict_dir, file))

        lumberjack.process_video(video=video_path, corrected = False)

        num_tree_pictures = len(os.listdir(output_dir))
        assert num_tree_pictures > 0

        expected_file = "pred_and_track_10sec_example.mp4"
        assert expected_file in os.listdir(predict_dir)

        for picture in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, picture))

        for video in os.listdir(predict_dir):
            os.remove(os.path.join(predict_dir, video))

        mock_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))
        os.remove(os.path.join(mock_dir, "corrected_10sec_example.mp4"))