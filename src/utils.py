"""Utility helpers for reproducibility, config loading, and data splits."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Any

import torch
import yaml
from torch.utils.data import DataLoader, random_split


class Config(dict):
    """Dictionary with dot-access for nested YAML configuration."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def dict_to_config(value: Any):
    """Recursively convert dictionaries into Config objects."""
    if isinstance(value, dict):
        return Config({k: dict_to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [dict_to_config(item) for item in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return it with dot access."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    return dict_to_config(cfg)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_split(dataset, batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader]:
    """Split a dataset into train and validation dataloaders."""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader