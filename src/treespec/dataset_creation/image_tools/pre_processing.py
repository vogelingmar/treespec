"""Functions to prepare images for dataset creation."""

import os
from pathlib import Path
from typing import Optional
import shutil
import imageio.v2 as imageio
import py360convert


def _copy_rectangle_image(
    rectangle_image_path: Path, output_color_image_filetype: str, output_image_face_dir_path: Path
) -> None:
    r"""Copies and renames a rectangle RGB image and names it based on its orientation.
    Args:
        rectangle_image_path: Path to the input rectangle RGB image.
        output_color_image_filetype: The file type for the output image (e.g., 'jpg', 'png').
        output_image_face_dir_path: Directory where the copied and renamed image will be saved.
    """
    parts = os.path.splitext(os.path.basename(rectangle_image_path))[0].split("_")
    idx = parts[-2]
    orientation_number = int(parts[-1])
    if orientation_number == 1:
        new_name = f"{idx}_rgb_left.{output_color_image_filetype}"
    else:
        new_name = f"{idx}_rgb_right.{output_color_image_filetype}"
    output_path = os.path.join(output_image_face_dir_path, new_name)
    shutil.copy2(rectangle_image_path, output_path)


def select_rectangle_images(
    input_rectangle_images_dir_path: Path,
    output_image_faces_dir_path: Path,
    input_rectangle_image_filetype: str,
    output_color_image_filetype: str,
    run_number: int,
) -> None:
    r"""Selects and copies left and right rectangle RGB images from the input directory to the output directory.

    Args:
        input_rectangle_images_dir_path: Directory containing the input rectangle RGB images.
        output_image_faces_dir_path: Directory where the copied and renamed images will be saved.
        input_rectangle_image_filetype: The file type of the input rectangle RGB images (e.g., 'jpg', 'png').
        output_color_image_filetype: The file type for the output images (e.g., 'jpg', 'png').
        run_number: The number of the recording run to filter images accordingly.
    """
    os.makedirs(output_image_faces_dir_path, exist_ok=True)
    rectangle_images_dir = os.listdir(input_rectangle_images_dir_path)

    for file in rectangle_images_dir:
        if file.endswith(f"{1}.{input_rectangle_image_filetype}") or file.endswith(
            f"{3}.{input_rectangle_image_filetype}"
        ):
            parts = os.path.splitext(file)[0].split("_")
            if len(parts) < 4:
                continue
            run = parts[0]
            if run.endswith(str(run_number)):
                rectangle_image_path = Path(os.path.join(input_rectangle_images_dir_path, file))
                _copy_rectangle_image(
                    rectangle_image_path=rectangle_image_path,
                    output_color_image_filetype=output_color_image_filetype,
                    output_image_face_dir_path=output_image_faces_dir_path,
                )

    print(f"Copied images to {output_image_faces_dir_path}")


def extract_pano_faces(  # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
    input_panos_dir_path: Path,
    output_faces_dir_path: Path,
    input_pano_filetype: str,
    output_face_filetype: str,
    run_number: int,
    apply_center_zoom: bool,
    name_filter: Optional[str] = None,
) -> None:
    r"""Extracts left and right faces from panoramic images in the input directory
    and saves them to the output directory.

    Args:
        input_panos_dir_path: Path to the directory containing input panoramic images.
        output_faces_dir_path: Path to the directory where extracted faces will be saved.
        input_pano_filetype: The file type of the input panoramic images (e.g., 'jpg', 'png').
        output_face_filetype: The file type for the output images (e.g., 'jpg', 'png').
        run_number: The number of the recording run to filter images accordingly.
        apply_center_zoom: Whether to zoom the faces to the center square (apply when using rectangle rgb images).
        name_filter: Optional filter to select specific types of images (e.g. 'segmentid', 'semanticclass').
            If left empty, type = 'rgb' is assumed.
    """
    os.makedirs(output_faces_dir_path, exist_ok=True)

    if name_filter is None or name_filter == "":
        image_type = "rgb"
    else:
        image_type = name_filter

    for file in sorted(os.listdir(input_panos_dir_path)):  # pylint: disable=too-many-nested-blocks
        if (
            image_type == "rgb"
            and file.endswith(f".{input_pano_filetype}")
            or image_type != "rgb"
            and file.endswith(f"_{name_filter}.{input_pano_filetype}")
        ):
            filename = os.path.splitext(file)[0]
            parts = filename.split("_")
            if len(parts) < 2:
                continue
            pano_run_number = parts[1]
            if pano_run_number.endswith(str(run_number)):
                pano = imageio.imread(os.path.join(input_panos_dir_path, file))
                pano_width = pano.shape[1]
                face_width = max(pano_width // 4, 500)

                cube_faces = py360convert.e2c(pano, face_w=face_width, cube_format="list", mode="nearest")

                image_number = int(parts[2])
                for i, face in enumerate(cube_faces):
                    if i in (1, 3):  # 1 = left, 3 = right
                        height, width = face.shape[:2]
                        if apply_center_zoom:
                            start_y, end_y = height // 4, 3 * height // 4
                            start_x, end_x = width // 4, 3 * width // 4
                            face = face[start_y:end_y, start_x:end_x]

                        # If semanticclass, extract red channel only
                        if image_type == "semanticclass":
                            # Ensure face has at least 3 channels
                            if face.ndim == 3 and face.shape[2] >= 1:
                                face = face[:, :, 0]  # R channel

                        filename_prefix = f"{image_number}_{image_type}"
                        face_label = "left" if i == 1 else "right"
                        output_path = os.path.join(
                            output_faces_dir_path, f"{filename_prefix}_{face_label}.{output_face_filetype}"
                        )
                        imageio.imwrite(output_path, face)

    print(f"Extracted {image_type} left and right faces from the panoramic images to {output_faces_dir_path}")
