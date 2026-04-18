"""
Patch extraction utilities for full-page handwriting images.
"""

import numpy as np
from typing import List

import config


def split_into_patches(
    image: np.ndarray,
    patch_size: int = 224,
    stride: int = 224,
    min_variance: float = None,
) -> List[np.ndarray]:
    """
    Extract non-overlapping patches from a handwriting image.

    Patches whose pixel variance is below `min_variance` are discarded —
    this filters out blank margins and near-white regions that carry no
    handwriting information and would dilute the risk estimate.

    Args:
        image:        H×W×3 uint8 NumPy array.
        patch_size:   Height and width of each square patch (pixels).
        stride:       Step between patch origins (pixels). Equal to
                      patch_size by default (no overlap).
        min_variance: Minimum pixel variance to keep a patch. Defaults
                      to config.PATCH_MIN_VARIANCE.

    Returns:
        List of H×W×3 uint8 patches that passed the variance filter.
    """
    if min_variance is None:
        min_variance = config.PATCH_MIN_VARIANCE

    import cv2

    h, w = image.shape[:2]

    # If the image is smaller than one patch, upscale it so we get at least one patch.
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w) * 1.1
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]

    patches: List[np.ndarray] = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            if patch.var() >= min_variance:
                patches.append(patch)

    # If variance filter discarded everything, return all patches without filtering.
    if not patches:
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patches.append(image[y : y + patch_size, x : x + patch_size])

    return patches
