"""Model definitions for image-based sentiment analysis."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Create a ResNet-18 backbone with a custom classification head."""

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
