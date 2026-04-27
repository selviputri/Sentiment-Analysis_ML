"""Training entry point for image-based sentiment analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .dataset import ImageSentimentDataset
    from .model import build_resnet18, get_device
    from .utils import load_config, set_seed
except ImportError:
    from dataset import ImageSentimentDataset
    from model import build_resnet18, get_device
    from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--save_dir", type=Path, default=Path("results/saved_models"))
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train_annotations", type=Path, help="Path to training annotations CSV")
    parser.add_argument("--val_annotations", type=Path, help="Path to validation annotations CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
            p=0.5,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    transforms_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    train_annotations = args.train_annotations or Path(cfg.data.train_annotations)
    val_annotations = args.val_annotations or Path(cfg.data.val_annotations)

    train_dataset = ImageSentimentDataset(
        annotations_file=train_annotations,
        root_dir=args.data_root,
        transform=transforms_train,
    )
    val_dataset = ImageSentimentDataset(
        annotations_file=val_annotations,
        root_dir=args.data_root,
        transform=transforms_val,
    )

    print(f"Train annotations: {train_annotations}")
    print(f"Validation annotations: {val_annotations}")
    print(
        "Train split counts:",
        train_dataset.annotations["sentiment"].value_counts().to_dict(),
    )
    print(
        "Validation split counts:",
        val_dataset.annotations["sentiment"].value_counts().to_dict(),
    )

    counts = train_dataset.annotations["sentiment_idx"].value_counts().reindex(
        range(cfg.model.num_classes), fill_value=0
    ).astype(float)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=getattr(cfg.training, 'num_workers', 2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=getattr(cfg.training, 'num_workers', 2),
    )

    device = get_device()

    raw_class_weights = counts.sum() / counts
    class_weights = torch.sqrt(torch.tensor(raw_class_weights.values, dtype=torch.float32))
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print(f"Using class weights: {class_weights.tolist()}")
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=getattr(cfg.training, "scheduler_patience", 2),
    )

    best_val_loss = float("inf")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.save_dir / "best.pth"
    patience = getattr(cfg.training, "patience", 4)
    no_improve = 0

    for epoch in range(1, cfg.training.epochs + 1):
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
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs.")

        if no_improve >= patience:
            print(f"Stopping early after {epoch} epochs.")
            break


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
