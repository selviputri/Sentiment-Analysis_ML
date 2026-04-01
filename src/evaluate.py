"""Evaluation utilities for the trained sentiment model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
from model import build_resnet18, get_device
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--annotations", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    dataset = ImageSentimentDataset(
        annotations_file=args.annotations,
        root_dir=args.data_root,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    device = get_device()
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    target_names = [INDEX_TO_SENTIMENT[i] for i in sorted(INDEX_TO_SENTIMENT)]
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
