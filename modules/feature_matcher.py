import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_features(img1_path, img2_path, output_path=None, max_matches=50):
    """
    Detect and match ORB keypoints between two drone images.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_path: Optional path to save visualization
        max_matches: Maximum number of matches to keep

    Returns:
        matches: List of cv2.DMatch objects
        kp1, kp2: Keypoints from image 1 and image 2
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB keypoint detection
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Brute-force matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

    # Draw matches using OpenCV for quick check
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    if output_path:
        cv2.imwrite(output_path, matched_img)
    else:
        cv2.imshow("Feature Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return matches, kp1, kp2


def plot_matches(img1, img2, kp1, kp2, matches, inliers=None, figsize=(15,8), save_path=None):
    """
    Visualize matches with matplotlib.

    Args:
        img1, img2: OpenCV images
        kp1, kp2: Keypoints
        matches: List of cv2.DMatch
        inliers: Optional boolean array marking inliers (True) / outliers (False)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Convert BGR -> RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if inliers is not None:
        matches_inliers = [m for m, inl in zip(matches, inliers) if inl]
        matches_outliers = [m for m, inl in zip(matches, inliers) if not inl]
    else:
        matches_inliers = matches
        matches_outliers = []

    # Draw inliers in green
    img_matches = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, matches_inliers, None,
        matchColor=(0,255,0), flags=2
    )

    # Draw outliers in red
    img_matches = cv2.drawMatches(
        img1_rgb, kp1, img2_rgb, kp2, matches_outliers, img_matches,
        matchColor=(255,0,0), flags=2
    )

    plt.figure(figsize=figsize)
    plt.imshow(img_matches)
    plt.axis('off')
    plt.title("Green: Inliers, Red: Outliers / Mismatches")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
