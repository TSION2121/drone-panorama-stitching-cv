import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from modules.feature_matcher import match_features
from modules.homography_estimator import estimate_homography
from modules.stitcher import blend_images

# Downscale factor for memory efficiency
DOWNSCALE_FACTOR = 0.05  # 5% of original size

# Output dirs for visualization
MATCH_DIR = "output/matches"
HEATMAP_DIR = "output/homography_heatmaps"
PANORAMA_DIR = "output/panorama_steps"
os.makedirs(MATCH_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(PANORAMA_DIR, exist_ok=True)

# Maximum panorama dimension to prevent OpenCV crash
MAX_DIM = 8000

def load_image_scaled(path, scale=DOWNSCALE_FACTOR):
    """Load image and downscale for memory efficiency."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def warp_images_safe(img1, img2, H):
    """
    Warp img1 into img2's coordinate system safely.
    Canvas is automatically sized to fit both images.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    all_corners = np.vstack((warped_corners, corners_img2))

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]

    width = min(xmax - xmin, MAX_DIM)
    height = min(ymax - ymin, MAX_DIM)

    translation_mat = np.array([[1, 0, translation[0]],
                                [0, 1, translation[1]],
                                [0, 0, 1]])

    result = cv2.warpPerspective(img1, translation_mat @ H, (width, height))

    # Prevent overflow if canvas was capped
    h2_c = min(h2, height - translation[1])
    w2_c = min(w2, width - translation[0])
    result[translation[1]:translation[1]+h2_c, translation[0]:translation[0]+w2_c] = img2[:h2_c, :w2_c]

    return result

def visualize_matches(kp1, kp2, matches, img1_path, img2_path, output_path):
    """Visualize feature matches using Matplotlib."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Matches: {os.path.basename(img1_path)} -> {os.path.basename(img2_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Match plot saved: {output_path}")

def visualize_homography(H, output_path, title="Homography Heatmap"):
    """Visualize a homography matrix as a heatmap."""
    plt.figure(figsize=(4,4))
    plt.imshow(H, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    for r in range(3):
        for c in range(3):
            plt.text(c, r, f"{H[r, c]:.2f}", color='white', ha='center', va='center')
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Homography heatmap saved: {output_path}")

def stitch_images(image_paths, output_path="output/final_panorama.jpg"):
    """Memory-safe stitching of multiple images with visualization."""
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images to stitch")

    os.makedirs("output", exist_ok=True)

    base_img = load_image_scaled(image_paths[0])

    for i in range(1, len(image_paths)):
        next_img = load_image_scaled(image_paths[i])

        matches, kp1, kp2 = match_features(
            img1_path=image_paths[i-1],
            img2_path=image_paths[i]
        )

        # Save match visualization
        match_plot_path = f"{MATCH_DIR}/matches_{i}.png"
        visualize_matches(kp1, kp2, matches, image_paths[i-1], image_paths[i], match_plot_path)

        # Homography estimation
        H, status = estimate_homography(kp1, kp2, matches)
        if H is None:
            print(f"[WARNING] Homography failed for {image_paths[i-1]} -> {image_paths[i]}")
            continue

        print(f"[INFO] Homography computed: {os.path.basename(image_paths[i-1])} -> {os.path.basename(image_paths[i])}")

        # Save homography heatmap
        heatmap_path = f"{HEATMAP_DIR}/homography_{i}.png"
        visualize_homography(H, heatmap_path, title=f"Homography {i}")

        # Warp images safely
        base_img = warp_images_safe(base_img, next_img, H)

        # Memory-efficient blending
        h, w = base_img.shape[:2]
        scale = 0.1  # 10% size for blending
        small_img = cv2.resize(base_img, (int(w*scale), int(h*scale)))
        small_img = blend_images(small_img, small_img)
        base_img = cv2.resize(small_img, (w, h))

        # Save intermediate panorama
        step_path = f"{PANORAMA_DIR}/panorama_step_{i}.jpg"
        cv2.imwrite(step_path, base_img)
        print(f"[INFO] Saved intermediate panorama: {step_path}")

    # Save final panorama
    cv2.imwrite(output_path, base_img)
    print(f"[SUCCESS] Final panorama saved at {output_path}")

if __name__ == "__main__":
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
