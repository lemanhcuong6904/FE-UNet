from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def build_transforms(image_size: tuple[int, int], augment: bool) -> A.Compose:
    height, width = image_size
    transforms: list[A.BasicTransform] = [
        A.Resize(height=height, width=width),
    ]
    if augment:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-20, 20),
                    border_mode=0,
                    fill=0,
                    fill_mask=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.4,
                ),
                #A.GaussNoise(p=0.2),
                #A.MotionBlur(blur_limit=3, p=0.15),
            ]
        )
    
    return A.Compose(transforms)


class PolypSegDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path = "polyp_data",
        split: str = "train",
        image_size: int | Tuple[int, int] = 350,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.image_dir = self.data_root / split / "images"
        self.mask_dir = self.data_root / split / "masks"
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.augment = augment
        self.transforms = build_transforms(image_size, augment)

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split folders: {self.image_dir} and/or {self.mask_dir}. "
                "Run split_polyp_data.py first."
            )

        self.image_paths = sorted(p for p in self.image_dir.iterdir() if p.is_file())
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        mask_tensor = (mask_tensor > 127.5).float()
        return image_tensor, mask_tensor, image_path.name


def build_polyp_loader(
    data_root: str | Path,
    split: str,
    batch_size: int = 12,
    image_size: int = 350,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    dataset = PolypSegDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        augment=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
