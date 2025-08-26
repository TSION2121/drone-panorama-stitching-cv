import cv2
import numpy as np

def estimate_homography(kp1, kp2, matches, reproj_thresh=4.0):
    # Extract matched keypoint coordinates
    ptsA = np.float32([kp1[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute homography with RANSAC
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
    return H, status
