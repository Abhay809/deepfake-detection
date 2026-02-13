"""
Generate sample real and fake images so you can run training without external data.
Real: natural-looking gradients + mild noise. Fake: different texture/blur to simulate artifacts.
"""

import os
import numpy as np
from PIL import Image

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_dir = os.path.join(root, "data", "real")
    fake_dir = os.path.join(root, "data", "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    size = 224
    n_real = 40
    n_fake = 40
    np.random.seed(42)

    # "Real" – smoother gradients, subtle noise (simulate natural photos)
    for i in range(n_real):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        r = 0.5 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + np.random.rand(size, size) * 0.1
        g = 0.4 + 0.35 * np.cos(3 * np.pi * X) + np.random.rand(size, size) * 0.1
        b = 0.5 + 0.25 * (X + Y) + np.random.rand(size, size) * 0.1
        img = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(real_dir, f"real_{i:03d}.png"))

    # "Fake" – more uniform/blurred patches + different noise (simulate blending/compression artifacts)
    for i in range(n_fake):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        # Slightly different frequency content and blockier pattern
        r = 0.5 + 0.2 * np.sin(4 * np.pi * X) * np.sin(4 * np.pi * Y) + np.random.rand(size, size) * 0.15
        g = 0.45 + 0.25 * np.cos(5 * np.pi * X * Y) + np.random.rand(size, size) * 0.15
        b = 0.55 + 0.2 * (X - Y) ** 2 + np.random.rand(size, size) * 0.15
        img = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
        # Simulate slight blur/compression
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.8)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(fake_dir, f"fake_{i:03d}.png"))

    print(f"Created {n_real} images in data/real/ and {n_fake} images in data/fake/")
    print("You can now run: python run_train.py")


if __name__ == "__main__":
    main()
