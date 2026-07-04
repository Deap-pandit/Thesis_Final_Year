import cv2
import numpy as np


def apply_exg(image: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply Excess Green (ExG) masking to isolate leaf pixels from background.

    Args:
        image: Input RGB image as a numpy array with shape (H, W, 3) and dtype uint8.
        threshold: Optional fixed threshold value for ExG. If None, use Otsu thresholding.

    Returns:
        masked_image: The original RGB image with background pixels blacked out.
        binary_mask: Binary leaf mask where leaf pixels are 255 and background is 0.
    """
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with shape (H, W, 3)")

    # Convert image to float in [0, 1]
    image_float = image.astype(np.float32) / 255.0
    r_channel = image_float[:, :, 0]
    g_channel = image_float[:, :, 1]
    b_channel = image_float[:, :, 2]

    # Excess Green index
    exg = 2.0 * g_channel - r_channel - b_channel
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold the ExG map
    if threshold is None:
        _, binary = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold_value = int(np.clip(threshold, 0, 255))
        _, binary = cv2.threshold(exg_norm, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological clean-up to remove speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Ensure leaf area is not empty
    if np.count_nonzero(binary) < 10:
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=2)

    masked_image = image.copy()
    masked_image[binary == 0] = 0

    return masked_image, binary
