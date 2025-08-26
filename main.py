from modules.feature_matcher import match_features
from modules.homography_estimator import estimate_homography
import cv2

if __name__ == "__main__":
    matches, kp1, kp2 = match_features(
        img1_path="data/drone1.jpg",
        img2_path="data/drone2.jpg",
        output_path="output/matches.jpg"
    )

    H, status = estimate_homography(kp1, kp2, matches)
    print(f"Homography matrix:\n{H}")

    # Warp image 1 to image 2's perspective
    img1 = cv2.imread("data/drone1.jpg")
    img2 = cv2.imread("data/drone2.jpg")
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    cv2.imwrite("output/aligned.jpg", result)
