import os

import cv2
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def canny_edge_count(patch):
    patch_cv = np.array(patch)[:, :, ::-1]
    edges = cv2.Canny(patch_cv, 50, 150)
    return np.sum(edges > 0)

def patch_img(img, step_size=112):
    img_width, img_height = img.size
    step_size = int(step_size)
    num_patches_x = (img_width + step_size - 1) // step_size
    num_patches_y = (img_height + step_size - 1) // step_size
    patch_list = []

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            x1 = j * step_size
            y1 = i * step_size
            x2 = min((j + 1) * step_size, img_width)
            y2 = min((i + 1) * step_size, img_height)
            patch = img.crop((x1, y1, x2, y2))
            patch_list.append((patch, canny_edge_count(patch)))

    patch_list.sort(key=lambda x: x[1])

    new_img, _ = patch_list[0]
    last_img, _ = patch_list[-1]

    new_img = new_img.resize((112, 112), Image.BILINEAR)
    last_img = last_img.resize((112, 112), Image.BILINEAR)

    return new_img, last_img
