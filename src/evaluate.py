"""Evaluation utilities for the trained sentiment model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import (
    LABEL_MAP,
    SentimentImageDataset,
    load_config,
    validate_config,
    load_annotations,
    split_data,
    get_transforms,
)
from model import build_resnet18, get_device


INDEX_TO_SENTIMENT = {v: k for k, v in LABEL_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    # Load full dataset and recreate same split logic used in training
    df = load_annotations(config)
    _, _, test_df = split_data(
        df=df,
        val_split=config["training"]["val_split"],
        test_split=config["training"]["test_split"],
        seed=config["training"]["seed"],
    )

    # Use evaluation transform only
    _, eval_transform = get_transforms(config)

    test_dataset = SentimentImageDataset(test_df, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config.get("dataloader", {}).get("num_workers", 0),
        pin_memory=config.get("dataloader", {}).get("pin_memory", False),
    )

    device = get_device()
    model = build_resnet18(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Support both full checkpoint dict and plain state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    target_names = [INDEX_TO_SENTIMENT[i] for i in sorted(INDEX_TO_SENTIMENT)]

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print("Confusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()