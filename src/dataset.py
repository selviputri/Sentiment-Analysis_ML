"""Dataset utilities for image-based sentiment analysis.

This module exposes `ImageSentimentDataset`, a thin wrapper around
preprocessed CSV metadata that maps image paths to integer sentiment labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SENTIMENT_TO_INDEX = {"negative": 0, "neutral": 1, "positive": 2}
INDEX_TO_SENTIMENT = {idx: label for label, idx in SENTIMENT_TO_INDEX.items()}


class ImageSentimentDataset(Dataset):
    """Simple dataset that expects a CSV file with image paths and sentiment labels.

    The CSV file must contain at least two columns: `image_path` and `sentiment`.
    Image paths are resolved relative to `root_dir`.
    """

    def __init__(
        self,
        annotations_file: Path | str,
        root_dir: Path | str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

        missing_columns = {"image_path", "sentiment"} - set(self.annotations.columns)
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {columns}")

        self.annotations["sentiment_idx"] = self.annotations["sentiment"].map(SENTIMENT_TO_INDEX)
        if self.annotations["sentiment_idx"].isna().any():
            raise ValueError("Unknown sentiment labels found in annotations file.")

    def __len__(self) -> int:  # pragma: no cover - trivial passthrough
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.annotations.iloc[idx]
        image_path = self.root_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["sentiment_idx"])
        return image, label

    @staticmethod
    def label_distribution(labels: List[int]) -> dict[int, int]:
        counts = pd.Series(labels).value_counts().to_dict()
        return {int(label): int(count) for label, count in counts.items()}
