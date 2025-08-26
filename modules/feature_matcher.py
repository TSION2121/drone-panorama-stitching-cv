# modules/feature_matcher.py

import cv2
import numpy as np

def match_features(img1_path, img2_path, output_path=None, max_matches=50):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors using brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None, flags=2)

    # Save or display
    if output_path:
        cv2.imwrite(output_path, matched_img)
    else:
        cv2.imshow("Feature Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return matches, kp1, kp2
