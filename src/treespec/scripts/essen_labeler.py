from PIL import Image
import numpy as np
import os
import cv2
import math

import py360convert
import imageio.v2 as imageio

# Paths to the images
basepath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

map_path = os.path.join(basepath, "io/helpers/cubemap.png")
texture_path = os.path.join(basepath, "io/helpers/pano_000037_000072_depth.png")
output_path = os.path.join(basepath, "io/pictures")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

img = imageio.imread(texture_path)
img = np.flip(img, axis=1)
cube_faces = py360convert.e2c(img, face_w=2048, cube_format='list')  # returns list of 6 faces

for i, face in enumerate(cube_faces):
    if i == 1 or i == 3:
        height, width = face.shape[:2]
        start_y, end_y = height // 4, 3 * height // 4
        start_x, end_x = width // 4, 3 * width // 4

        cropped_face = face[start_y:end_y, start_x:end_x]

        upsampled_face = cv2.resize(cropped_face, (2048, 2048), interpolation=cv2.INTER_CUBIC)

        imageio.imwrite(os.path.join(output_path, f"cube_face_{i}.png"), upsampled_face)
