# main.py

from modules.feature_matcher import match_features

if __name__ == "__main__":
    matches, kp1, kp2 = match_features(
        img1_path="data/drone1.jpg",
        img2_path="data/drone2.jpg",
        output_path="output/matches.jpg"
    )
    print(f"Total matches found: {len(matches)}")
