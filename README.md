# Drone Panorama Stitching (CV Final Project)

This project stitches overlapping drone images into a seamless panorama using feature matching, RANSAC, and warping techniques.

## Phase 1 — ORB Feature Detection & Matching

The goal of Phase 1 is to find visual “landmarks” that appear in both drone images so we can align them into a panorama.

**Steps:**
1. **Load & Preprocess** — Convert both images to grayscale for feature detection.
2. **Detect Keypoints (ORB)** — Identify distinctive points like runway markings or building corners.
3. **Match Descriptors** — Compare feature fingerprints between images using a brute-force matcher.
4. **Sort & Visualize** — Draw the top matches and save them to `output/matches.jpg`.

These matched points are the “anchor points” we’ll use in Phase 2 to calculate the homography and align the images.


## Tools
- Python 3.11
- OpenCV
- IntelliJ IDEA

## Structure
- `data/`: input drone images
- `modules/`: feature matching, homography, stitching
- `output/`: final panoramas
- `main.py`: pipeline entry point

## How to Run
```bash
python main.py
