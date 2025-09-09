import cv2
import numpy as np
import os
from modules.feature_matcher import match_features
from modules.homography_estimator import estimate_homography
from modules.stitcher import blend_images

# Downscale factor for high-res images
DOWNSCALE_FACTOR = 0.25  # 25% of original size

def load_image_scaled(path, scale=DOWNSCALE_FACTOR):
    """Load an image and downscale to reduce memory usage."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def warp_images(img1, img2, H):
    """Warp img1 into the coordinate system of img2 using homography H."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    all_corners = np.vstack((warped_corners, corners_img2))

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    translation_mat = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    result = cv2.warpPerspective(img1, translation_mat @ H, (xmax-xmin, ymax-ymin))
    result[translation[1]:h2+translation[1], translation[0]:w2+translation[0]] = img2
    return result

def stitch_images(image_paths, output_path="output/final_panorama.jpg"):
    """Stitch a sequence of images into a panorama (memory-safe)."""
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    base_img = load_image_scaled(image_paths[0])

    for i in range(1, len(image_paths)):
        next_img = load_image_scaled(image_paths[i])

        # Step 1: Feature Matching
        matches, kp1, kp2 = match_features(
            img1_path=image_paths[i-1],
            img2_path=image_paths[i],
            output_path=f"output/matches_{i}.jpg"
        )

        # Step 2: Homography Estimation
        H, status = estimate_homography(kp1, kp2, matches)
        if H is None:
            print(f"[WARNING] Homography failed for {image_paths[i-1]} -> {image_paths[i]}")
            continue

        print(f"[INFO] Homography computed for {os.path.basename(image_paths[i-1])} -> {os.path.basename(image_paths[i])}")

        # Step 3: Warp
        base_img = warp_images(base_img, next_img, H)

        # Step 4: Pad next image to match base_img size
        h_base, w_base = base_img.shape[:2]
        h_next, w_next = next_img.shape[:2]
        next_img_padded = np.zeros_like(base_img)
        next_img_padded[:h_next, :w_next] = next_img

        # Step 5: Blend
        base_img = blend_images(base_img, next_img_padded)

        # Save intermediate panorama
        cv2.imwrite(f"output/panorama_step_{i}.jpg", base_img)
        print(f"[INFO] Saved intermediate panorama: output/panorama_step_{i}.jpg")

    # Save final panorama
    cv2.imwrite(output_path, base_img)
    print(f"[SUCCESS] Final panorama saved at {output_path}")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Explicit list of 10 drone images
    image_paths = [
        "data/drone1.jpg",
        "data/drone2.jpg",
        "data/drone3.jpg",
        "data/drone4.jpg",
        "data/drone5.jpg",
        "data/drone6.jpg",
        "data/drone7.jpg",
        "data/drone8.jpg",
        "data/drone9.jpg",
        "data/drone10.jpg"
    ]

    print(f"[INFO] Found {len(image_paths)} images: {image_paths}")
    stitch_images(image_paths)
