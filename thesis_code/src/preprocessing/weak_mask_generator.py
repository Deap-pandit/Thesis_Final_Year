from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def generate_weak_mask(image: Image.Image | np.ndarray, lower_hsv: tuple[int, int, int] = (0, 40, 40), upper_hsv: tuple[int, int, int] = (179, 255, 255)) -> np.ndarray:
    """Create a weak lesion mask using HSV thresholding as a placeholder.

    This is a stopgap method for thesis development when no manually annotated lesion masks exist.
    Replace this with true pixel annotations from CVAT/LabelMe or a stronger domain-specific detector
    if you later acquire a segmentation dataset.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)

    if image_np.ndim != 3:
        raise ValueError("Expected a 3-channel RGB image")

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)
