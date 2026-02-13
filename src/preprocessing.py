"""
Preprocessing pipeline for DeepFake detection.
Handles image loading, frequency-domain features (FFT), and texture consistency cues.
"""

import numpy as np
import cv2
from PIL import Image
from scipy import fft
from typing import Tuple, Optional
import torch
from torchvision import transforms


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_frequency_features(img: np.ndarray) -> np.ndarray:
    """
    Extract frequency-domain features to capture compression/artifact distortions.
    Deepfakes often show anomalies in high-frequency components.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    f = fft.fft2(gray)
    fshift = fft.fftshift(f)
    magnitude = np.abs(fshift)
    # Log scale for stability
    magnitude = np.log1p(magnitude)
    # Resize to fixed size for CNN input
    magnitude = cv2.resize(magnitude, (56, 56))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)


def compute_texture_features(img: np.ndarray) -> np.ndarray:
    """
    Simple texture consistency cues via local std (blending errors often change local variance).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel_size = 5
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))
    local_std = cv2.resize(local_std, (56, 56))
    local_std = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)
    return local_std.astype(np.float32)


def preprocess_for_model(
    img: np.ndarray,
    image_size: int = 224,
    use_frequency: bool = True,
    use_texture: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Preprocess image for CNN: resize, normalize, optionally stack frequency/texture.
    Returns (rgb_tensor, optional_aux_tensor).
    """
    rgb = cv2.resize(img, (image_size, image_size))
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    extra_channels = []
    if use_frequency:
        freq = compute_frequency_features(img)
        extra_channels.append(freq)
    if use_texture:
        tex = compute_texture_features(img)
        extra_channels.append(tex)

    if extra_channels:
        aux = np.stack(extra_channels, axis=0)  # (2, 56, 56) or (1, 56, 56)
        aux_tensor = torch.from_numpy(aux).float()
        return rgb_tensor, aux_tensor
    return rgb_tensor, None


# Standard ImageNet-style normalization for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, augment: bool = True):
    """Training transforms with optional augmentation."""
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224):
    """Validation/test transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
