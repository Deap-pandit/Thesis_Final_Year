from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset for 19-class or discovered-class leaf disease classification."""

    def __init__(self, manifest_path: str, split: str, image_size: int = 224, train: bool = False, class_names: list[str] | None = None):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.train = train

        self.class_names = class_names or sorted(self.manifest["label"].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

        self.transform = self._build_transform(train)

    def _build_transform(self, train: bool):
        if train:
            return transforms.Compose(
                [
                    transforms.RandomRotation(25),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        image_path = Path(row["filepath"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label_idx = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
        return image_tensor, label_idx
