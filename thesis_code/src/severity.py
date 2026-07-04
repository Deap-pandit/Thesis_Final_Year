from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image


def compute_severity(mask: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Compute severity percentage as diseased_pixel_count / leaf_pixel_count * 100."""
    diseased_pixels = int(np.sum(mask > 0.5))
    leaf_pixels = int(np.sum(leaf_mask > 0))
    if leaf_pixels == 0:
        return 0.0
    return (diseased_pixels / leaf_pixels) * 100.0


def overlay_severity(image: np.ndarray, disease_mask: np.ndarray, severity_percent: float, output_path: str | None = None) -> np.ndarray:
    """Create an overlay that highlights diseased regions on the original leaf image."""
    overlay = image.copy()
    red_mask = np.zeros_like(image, dtype=np.uint8)
    red_mask[:, :, 2] = 255
    overlay[disease_mask > 0.5] = cv2.addWeighted(overlay[disease_mask > 0.5], 0.5, red_mask[disease_mask > 0.5], 0.5, 0)
    cv2.putText(overlay, f"Severity: {severity_percent:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay
