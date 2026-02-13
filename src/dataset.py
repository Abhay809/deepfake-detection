"""
Dataset classes for real vs deepfake image classification.
"""

import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .preprocessing import (
    get_train_transforms,
    get_val_transforms,
    compute_frequency_features,
    compute_texture_features,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class DeepFakeDataset(Dataset):
    """
    Dataset of images from real/ and fake/ (or authentic/ and manipulated/) folders.
    Optionally includes frequency and texture auxiliary channels.
    """

    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        image_size: int = 224,
        is_train: bool = True,
        augment: bool = True,
        use_frequency: bool = True,
        use_texture: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.image_size = image_size
        self.use_frequency = use_frequency
        self.use_texture = use_texture
        self.samples = []  # (path, label): 0 = real, 1 = fake

        for path in Path(real_dir).rglob("*"):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self.samples.append((str(path), 0))
        for path in Path(fake_dir).rglob("*"):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self.samples.append((str(path), 1))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {real_dir} or {fake_dir}. "
                "Create folders data/real/ and data/fake/ and add images."
            )

        if transform is not None:
            self.transform = transform
        else:
            t = get_train_transforms(image_size, augment) if is_train else get_val_transforms(image_size)
            self.transform = t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        rgb = self.transform(Image.fromarray(img))

        aux_list = []
        if self.use_frequency:
            aux_list.append(compute_frequency_features(img))
        if self.use_texture:
            aux_list.append(compute_texture_features(img))

        if aux_list:
            aux = np.stack(aux_list, axis=0)
            aux = torch.from_numpy(aux).float()
            return rgb, aux, torch.tensor(label, dtype=torch.long)
        return rgb, torch.tensor(label, dtype=torch.long)


def collate_with_aux(batch):
    """Collate batch when samples are (rgb, aux, label)."""
    if len(batch[0]) == 3:
        rgb = torch.stack([b[0] for b in batch])
        aux = torch.stack([b[1] for b in batch])
        labels = torch.stack([b[2] for b in batch])
        return rgb, aux, labels
    rgb = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return rgb, None, labels
