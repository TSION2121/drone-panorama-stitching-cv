import cv2
import numpy as np

def blend_images(aligned_img, base_img):
    """
    Memory-efficient blending for large panoramas.
    Only blends overlapping region with linear mask.
    """
    aligned = aligned_img.astype(np.float16)
    base = base_img.astype(np.float16)

    # Find overlapping region
    overlap_mask = (base.sum(axis=2) > 0) & (aligned.sum(axis=2) > 0)

    blended = base.copy()

    # Only blend where overlap exists
    for c in range(3):  # for each color channel
        blended[..., c][overlap_mask] = (
                0.5 * aligned[..., c][overlap_mask] +
                0.5 * base[..., c][overlap_mask]
        )

    # Fill non-overlapping areas from aligned image
    non_overlap = (aligned.sum(axis=2) > 0) & (~overlap_mask)
    for c in range(3):
        blended[..., c][non_overlap] = aligned[..., c][non_overlap]

    return np.clip(blended, 0, 255).astype(np.uint8)
