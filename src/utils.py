"""Utility helpers for reproducibility, config loading, and data splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch.utils.data import DataLoader, random_split


@dataclass
class Config:
    data: object
    model: object
    training: object


def load_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    return Config(**{k: SimpleNamespace(**v) for k, v in cfg.items()})


class SimpleNamespace(dict):
    """Allow dot access to dictionary attributes."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_split(dataset, batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader]:
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
