from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class DenseNetClassifier(nn.Module):
    """DenseNet-121 transfer learning classifier for leaf disease recognition."""

    def __init__(self, num_classes: int, fine_tune_mode: str = "freeze_head"):
        super().__init__()
        self.num_classes = num_classes
        self.fine_tune_mode = fine_tune_mode
        try:
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        except Exception:
            self.backbone = models.densenet121(weights=None)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

        self._configure_finetuning()

    def _configure_finetuning(self) -> None:
        if self.fine_tune_mode == "freeze_head":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.fine_tune_mode == "fine_tune_last_block":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.features.denseblock4.parameters():
                param.requires_grad = True
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unsupported fine_tune_mode: {self.fine_tune_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
