"""Evaluation utilities for the trained sentiment model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
    from .model import build_resnet18, get_device
    from .utils import load_config
except ImportError:
    from dataset import ImageSentimentDataset, INDEX_TO_SENTIMENT
    from model import build_resnet18, get_device
    from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--annotations", type=Path, help="Path to annotations CSV for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    annotations_path = args.annotations or Path(cfg.data.test_annotations)
    print(f"Evaluating on annotations: {annotations_path}")

    dataset = ImageSentimentDataset(
        annotations_file=annotations_path,
        root_dir=args.data_root,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    device = get_device()
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=False)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except KeyError:
        print("Error: 'model_state_dict' key not found in checkpoint")
        return
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

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
