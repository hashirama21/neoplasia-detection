"""
Dataset and endoscopy-specific augmentations for RARE26.
Augmentations simulate real clinical acquisition variability:
- Specular highlights (endoscope light reflections)
- NBI mode simulation (Narrow Band Imaging color shift)
- Motion blur (camera movement)
- Domain shift robustness (Olympus train → Fuji/Pentax test)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


class SpecularHighlightSimulation:
    """Simulates endoscope light reflections — major FP source for naive models."""

    def __init__(self, prob: float = 0.3, max_spots: int = 3):
        self.prob = prob
        self.max_spots = max_spots

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        import numpy as np
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        n_spots = random.randint(1, self.max_spots)
        for _ in range(n_spots):
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            radius = random.randint(5, 25)
            intensity = random.uniform(0.6, 1.0)
            y_idx, x_idx = np.ogrid[:h, :w]
            mask = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) <= radius ** 2
            arr[mask] = arr[mask] * (1 - intensity) + 255 * intensity
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


class NBISimulation:
    """Simulates Narrow Band Imaging — reduces red channel, enhances green/blue."""

    def __init__(self, prob: float = 0.2):
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        import numpy as np
        arr = np.array(img, dtype=np.float32)
        # NBI reduces red channel and enhances vascular patterns in green/blue
        factor_r = random.uniform(0.3, 0.6)
        factor_g = random.uniform(1.0, 1.3)
        factor_b = random.uniform(0.8, 1.1)
        arr[:, :, 0] = (arr[:, :, 0] * factor_r).clip(0, 255)
        arr[:, :, 1] = (arr[:, :, 1] * factor_g).clip(0, 255)
        arr[:, :, 2] = (arr[:, :, 2] * factor_b).clip(0, 255)
        return Image.fromarray(arr.astype(np.uint8))


class MotionBlur:
    """Simulates endoscope motion blur from camera movement."""

    def __init__(self, prob: float = 0.2, max_kernel: int = 7):
        self.prob = prob
        self.max_kernel = max_kernel

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        kernel_size = random.choice([3, 5, 7])
        kernel_size = min(kernel_size, self.max_kernel)
        return img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))


class GaussianNoise:
    """Additive Gaussian noise — simulates sensor noise variability across scopes."""

    def __init__(self, prob: float = 0.2, std: float = 0.05):
        self.prob = prob
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        arr = np.array(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0, self.std, arr.shape).astype(np.float32)
        arr = (arr + noise).clip(0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))


def build_train_transforms(cfg) -> transforms.Compose:
    aug = cfg.augmentation.train
    norm = cfg.normalize
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.image_size,
            scale=tuple(aug.random_resized_crop.scale),
            ratio=tuple(aug.random_resized_crop.ratio),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(aug.horizontal_flip),
        transforms.RandomVerticalFlip(aug.vertical_flip),
        transforms.RandomRotation(aug.rotation_degrees),
        transforms.ColorJitter(
            brightness=aug.color_jitter.brightness,
            contrast=aug.color_jitter.contrast,
            saturation=aug.color_jitter.saturation,
            hue=aug.color_jitter.hue,
        ),
        SpecularHighlightSimulation(prob=aug.specular_highlight_prob),
        NBISimulation(prob=aug.nbi_simulation_prob),
        MotionBlur(prob=aug.motion_blur_prob),
        GaussianNoise(prob=aug.gaussian_noise_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm.mean, std=norm.std),
    ])


def build_val_transforms(cfg) -> transforms.Compose:
    norm = cfg.normalize
    return transforms.Compose([
        transforms.Resize(cfg.augmentation.val.resize,
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.augmentation.val.center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm.mean, std=norm.std),
    ])


class Rare26Dataset(Dataset):
    """
    Dataset for RARE26 challenge.
    CSV format: image_path,label (0=normal, 1=neoplasia)
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        root_dir: Optional[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None

        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert "label" in self.df.columns, "CSV must have 'label' column"

        self.labels = self.df["label"].values
        self.image_paths = self.df["image_path"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        if self.root_dir:
            img_path = self.root_dir / img_path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"image": image, "label": label, "path": str(img_path)}

    def get_class_weights(self, target_pos_ratio: float = 0.15) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler.

        target_pos_ratio controls the fraction of positives drawn per epoch.
        0.15 gives the model enough positive exposure without pushing the
        training prior far from the 1% test prevalence. Avoid 0.5 (full
        balance): it causes the model to predict positive for everything at
        inference time.
        """
        n_pos = int(self.labels.sum())
        n_neg = len(self.labels) - n_pos
        w_pos = target_pos_ratio / n_pos if n_pos > 0 else 1.0
        w_neg = (1.0 - target_pos_ratio) / n_neg if n_neg > 0 else 1.0
        weights = np.where(self.labels == 1, w_pos, w_neg)
        return torch.from_numpy(weights.astype(np.float32))
