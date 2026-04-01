"""Training entry point for image-based sentiment analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageSentimentDataset
from model import build_resnet18, get_device
from utils import load_config, set_seed, train_val_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--save_dir", type=Path, default=Path("results/saved_models"))
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    dataset = ImageSentimentDataset(
        annotations_file=cfg.data.annotations,
        root_dir=args.data_root,
        transform=transforms_train,
    )

    train_loader, val_loader = train_val_split(
        dataset,
        batch_size=cfg.training.batch_size,
        val_split=cfg.training.val_split,
    )

    device = get_device()
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    best_val_loss = float("inf")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.save_dir / "best.pth"

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def evaluate(model, dataloader, criterion, device) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


if __name__ == "__main__":
    main()
