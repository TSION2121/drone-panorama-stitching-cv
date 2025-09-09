import cv2
import numpy as np
import os
import pickle
from modules.feature_matcher import match_features
from modules.homography_estimator import estimate_homography
from modules.stitcher import blend_images
import matplotlib.pyplot as plt

LOWRES_FACTOR = 0.05  # Low-res for homography
MAX_PANORAMA_SIZE = 2000  # Maximum dimension for high-res stitching

def load_image_scaled(path, scale=1.0):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def warp_images_safe(img1, img2, H, max_size=MAX_PANORAMA_SIZE):
    """Warp img1 into img2's coordinate system safely with size limit."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Warp corners
    corners = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.vstack((warped_corners, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)))

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]
    width = min(xmax - xmin, max_size)
    height = min(ymax - ymin, max_size)

    translation_mat = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])
    result = cv2.warpPerspective(img1, translation_mat @ H, (width, height))

    # Crop img2 if too big
    h_crop = min(h2, height - translation[1])
    w_crop = min(w2, width - translation[0])
    result[translation[1]:translation[1]+h_crop, translation[0]:translation[0]+w_crop] = img2[:h_crop, :w_crop]

    return result

def compute_lowres_homographies(image_paths, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    homographies = []

    for i in range(1, len(image_paths)):
        img1 = load_image_scaled(image_paths[i-1], scale=LOWRES_FACTOR)
        img2 = load_image_scaled(image_paths[i], scale=LOWRES_FACTOR)

        matches, kp1, kp2 = match_features(
            img1_path=image_paths[i-1],
            img2_path=image_paths[i],
            output_path=f"{output_dir}/matches_plot_{i}.png"
        )

        H, status = estimate_homography(kp1, kp2, matches)
        if H is None:
            raise RuntimeError(f"Homography failed for {image_paths[i-1]} -> {image_paths[i]}")

        homographies.append(H)
        print(f"[INFO] Homography computed: {os.path.basename(image_paths[i-1])} -> {os.path.basename(image_paths[i])}")

    # Save homographies
    with open(f"{output_dir}/homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)
    print(f"[INFO] Saved {len(homographies)} homographies to {output_dir}/homographies.pkl")
    return homographies

def stitch_highres(image_paths, homographies, output_dir="output"):
    base_img = load_image_scaled(image_paths[0], scale=1.0)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, len(image_paths)):
        next_img = load_image_scaled(image_paths[i], scale=1.0)
        H = homographies[i-1]
        base_img = warp_images_safe(base_img, next_img, H)
        # Optional blending
        base_img = blend_images(base_img, base_img)
        step_path = f"{output_dir}/panorama_highres_step_{i}.jpg"
        cv2.imwrite(step_path, base_img)
        print(f"[INFO] Saved intermediate high-res panorama: {step_path}")

    final_path = f"{output_dir}/final_panorama_highres.jpg"
    cv2.imwrite(final_path, base_img)
    print(f"[SUCCESS] Final high-res panorama saved at {final_path}")

if __name__ == "__main__":
    image_paths = [f"data/drone{i}.jpg" for i in range(1, 11)]
    print(f"[INFO] Found {len(image_paths)} images: {image_paths}")

    # Step 1: Compute low-res homographies and save matches
    homographies = compute_lowres_homographies(image_paths)

    # Step 2: Stitch high-res images safely
    stitch_highres(image_paths, homographies)
