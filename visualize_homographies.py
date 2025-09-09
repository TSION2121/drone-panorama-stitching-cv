import matplotlib.pyplot as plt
import pickle
import os

# Load homographies
with open("output/homographies.pkl", "rb") as f:
    homographies = pickle.load(f)

os.makedirs("output/homography_heatmaps", exist_ok=True)

for i, H in enumerate(homographies, start=1):
    plt.figure(figsize=(4,4))
    plt.imshow(H, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(f"Homography {i}")
    # Show numeric values
    for r in range(3):
        for c in range(3):
            plt.text(c, r, f"{H[r, c]:.2f}", color='white', ha='center', va='center')
    plt.savefig(f"output/homography_heatmaps/homography_{i}.png")
    plt.close()

print(f"[INFO] Saved {len(homographies)} homography heatmaps in output/homography_heatmaps/")
