"""
Optional: Download a small sample dataset for quick testing.
Uses Kaggle Face Forensics or similar public deepfake datasets if available.
Otherwise, creates minimal placeholder structure and prints instructions.
"""

import os
import sys

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_dir = os.path.join(root, "data", "real")
    fake_dir = os.path.join(root, "data", "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Check for existing images
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    n_real = sum(1 for f in os.listdir(real_dir) if f.lower().endswith(exts))
    n_fake = sum(1 for f in os.listdir(fake_dir) if f.lower().endswith(exts))

    if n_real > 0 or n_fake > 0:
        print(f"Found {n_real} real and {n_fake} fake images. Ready for training.")
        return

    print("No images in data/real/ or data/fake/ yet.")
    print("\nTo train the model, add images:")
    print("  - data/real/  → authentic, unmanipulated face images")
    print("  - data/fake/  → deepfake or AI-generated face images")
    print("\nPublic datasets you can use:")
    print("  - Face Forensics++: https://github.com/ondyari/FaceForensics")
    print("  - Celeb-DF: https://github.com/yuezunliu-ceb/Celeb-DF")
    print("  - DFDC (Kaggle): https://www.kaggle.com/c/deepfake-detection-challenge")
    print("\nFor a quick test, place at least ~20 images in each folder.")


if __name__ == "__main__":
    main()
