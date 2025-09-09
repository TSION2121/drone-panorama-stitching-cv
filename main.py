from modules.feature_matcher import match_features
from modules.homography_estimator import estimate_homography
from modules.stitcher import blend_images
import cv2
import os

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Phase 1: Feature Matching
    matches, kp1, kp2 = match_features(
        img1_path="data/drone1.jpg",
        img2_path="data/drone2.jpg",
        output_path="output/matches.jpg"
    )

    # Phase 2: Homography Estimation
    H, status = estimate_homography(kp1, kp2, matches)
    print(f"Homography matrix:\n{H}")

    img1 = cv2.imread("data/drone1.jpg")
    img2 = cv2.imread("data/drone2.jpg")

    # Warp image 1 into image 2’s perspective
    warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    warped_img1[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imwrite("output/aligned.jpg", warped_img1)

    # Crop to base image size (optional step for visualization)
    cropped_warped = warped_img1[:, :img2.shape[1]]

    # Phase 3: Blend and save final panorama
    panorama = blend_images(cropped_warped, img2)
    cv2.imwrite("output/panorama.jpg", panorama)

    print("[SUCCESS] Two-image panorama created at output/panorama.jpg")
