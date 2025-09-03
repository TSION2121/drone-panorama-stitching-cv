import cv2
import numpy as np


"""
match_features.py

Phase 1: Detect and match ORB keypoints between two drone images.

Returns:
    matches: List of cv2.DMatch objects
    kp1, kp2: Keypoints from image 1 and image 2
"""

def match_features(img1_path, img2_path, output_path="output/matches.jpg", max_matches=50):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    cv2.imwrite(output_path, match_img)

    return matches, kp1, kp2
