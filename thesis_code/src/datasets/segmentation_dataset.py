from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.preprocessing.weak_mask_generator import generate_weak_mask


class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for weakly supervised lesion segmentation."""

    def __init__(self, manifest_path: str, split: str, image_size: int = 224, train: bool = False):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        self.transform = self._build_transform(train)

    def _build_transform(self, train: bool):
        if train:
            return transforms.Compose(
                [
                    transforms.RandomRotation(25),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        image_path = Path(row["filepath"])
        image = Image.open(image_path).convert("RGB")
        mask = generate_weak_mask(image)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_tensor = transforms.functional.to_tensor(
            transforms.Resize((self.image_size, self.image_size))(mask_image)
        ).float()
        image_tensor = self.transform(image)
        return image_tensor, mask_tensor
