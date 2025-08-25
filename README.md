feature_matcher.py: Load images → detect keypoints → match descriptors → visualize matches

homography_estimator.py: Use matched keypoints → apply RANSAC → compute homography matrix

stitcher.py: Warp one image → blend with the other → export final panorama

main.py: Orchestrates the full pipeline step by step
