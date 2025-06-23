import pytest
import os
from treespec.utils.image_tools import select_rgb_images, extract_pano_faces, extract_trees

# TODO: finish test and according mock data
testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_select_rgb_images():
    input_dir = os.path.join(testpath, "mock/essen")
    output_dir = os.path.join(testpath, "mock/temp/pictures")
    image_type = "jpg"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    select_rgb_images(input_dir, output_dir, image_type)

    output = os.listdir(output_dir)
    assert len(output) == 32
    for file in output:
        assert file.endswith("_rgb_left.jpg") or file.endswith("_rgb_right.jpg")
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(output_dir, file))


def test_extract_pano_faces():
    input_dir = os.path.join(testpath, "mock/essen/run_40")
    output_dir = os.path.join(testpath, "mock/temp/pictures")
    input_type = "tif"
    output_type = "png"
    run = "40"
    filter = "segmentid"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    extract_pano_faces(input_dir, output_dir, input_type, output_type, run, filter)

    output = os.listdir(output_dir)
    assert len(output) > 0
    for file in output:
        assert file.endswith(f"{filter}_left.{output_type}") or file.endswith(f"{filter}_right.{output_type}")
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(output_dir, file))


def test_extract_trees():
    segmentid_dir = os.path.join(testpath, "mock/temp/pictures")
    color_dir = "mock/temp/pictures"
    output_dir = "mock/temp/pictures"
    tree_attributes_dict = "kp"
    cover = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    extract_trees(segmentid_dir, color_dir, output_dir, tree_attributes_dict, cover)

    output = os.listdir(output_dir)
    assert len(output) > 0
    for file in output:
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(output_dir, file))
