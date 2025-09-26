import pytest
import os
import shutil
from treespec.dataset_creation.image_tools.pre_processing import (
    select_rectangle_images,
    extract_pano_faces,
)

dataset_creation_mock_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mock")


def test_select_rectangle_images():
    """Tests the select_rgb_images function."""
    input_rectangle_images_dir_path = os.path.join(dataset_creation_mock_dir_path, "images", "rectangle_color_images")
    output_image_faces_dir_path = os.path.join(dataset_creation_mock_dir_path, "temp", "pictures")
    input_rectangle_image_filetype = "jpg"
    output_color_image_filetype = "png"
    run_number = 70

    shutil.rmtree(output_image_faces_dir_path, ignore_errors=True)

    select_rectangle_images(
        input_rectangle_images_dir_path=input_rectangle_images_dir_path,
        output_image_faces_dir_path=output_image_faces_dir_path,
        input_rectangle_image_filetype=input_rectangle_image_filetype,
        output_color_image_filetype=output_color_image_filetype,
        run_number=run_number,
    )

    output = os.listdir(output_image_faces_dir_path)
    assert len(output) > 0
    for file in output:
        assert file.endswith(f"_rgb_left.{output_color_image_filetype}") or file.endswith(
            f"_rgb_right.{output_color_image_filetype}"
        )
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()

    shutil.rmtree(output_image_faces_dir_path, ignore_errors=True)


def test_extract_pano_faces():
    """Tests the extract_pano_faces function for different filters and image types."""
    input_panos_dir_path = os.path.join(dataset_creation_mock_dir_path, "images", "pano_segmentid_images")
    output_faces_dir_path = os.path.join(dataset_creation_mock_dir_path, "temp", "pano_faces")
    input_pano_filetype = "tif"
    output_face_filetype = "png"
    run_number = 70
    apply_center_crop = False
    name_filter = "segmentid"

    # test for segmentid images
    shutil.rmtree(output_faces_dir_path, ignore_errors=True)

    extract_pano_faces(
        input_panos_dir_path=input_panos_dir_path,
        output_faces_dir_path=output_faces_dir_path,
        input_pano_filetype=input_pano_filetype,
        output_face_filetype=output_face_filetype,
        run_number=run_number,
        apply_center_zoom=apply_center_crop,
        name_filter=name_filter,
    )

    output = os.listdir(output_faces_dir_path)
    assert len(output) > 0
    for file in output:
        assert file.endswith(f"{name_filter}_left.{output_face_filetype}") or file.endswith(
            f"{name_filter}_right.{output_face_filetype}"
        )
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()

    shutil.rmtree(output_faces_dir_path, ignore_errors=True)
