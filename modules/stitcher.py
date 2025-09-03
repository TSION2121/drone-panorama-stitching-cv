import cv2
import numpy as np

def blend_images(aligned_img, base_img):
    aligned = aligned_img.astype(np.float32)
    base = base_img.astype(np.float32)

    # Create a left-to-right blending mask
    mask = np.zeros_like(base, dtype=np.float32)
    mask[:, :base.shape[1]//2] = 1.0
    mask[:, base.shape[1]//2:] = 0.0
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    # Blend using the mask
    blended = aligned * (1 - mask) + base * mask
    return np.clip(blended, 0, 255).astype(np.uint8)
