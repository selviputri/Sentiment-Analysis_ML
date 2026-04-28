"""Visualization utilities for model results and error analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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


def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """Plot and save confusion matrix as heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
    )
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.title('Confusion Matrix - Sentiment Classification')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


def plot_class_distribution(annotations_df, save_path=None):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    counts = annotations_df['sentiment'].value_counts()
    axes[0].bar(counts.index, counts.values, color=['#d62728', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Class Distribution (Counts)')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_xlabel('Sentiment')
    
    # Pie chart
    percentages = counts.values / counts.sum() * 100
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    plt.show()


def plot_metrics_by_class(report_dict, save_path=None):
    """Plot precision, recall, F1 by class."""
    classes = [k for k in report_dict.keys() if k != 'accuracy']
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report_dict[cls].get(metric, 0) for cls in classes]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics by Sentiment Class')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics chart to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--annotations", type=Path, help="Path to annotations CSV")
    parser.add_argument("--output_dir", type=Path, default=Path("results/visualizations"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config(args.config)
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])
    
    annotations_path = args.annotations or Path(cfg.data.test_annotations)
    dataset = ImageSentimentDataset(
        annotations_file=annotations_path,
        root_dir=args.data_root,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)
    
    # Load model
    device = get_device()
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Get predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    
    # Generate metrics
    target_names = [INDEX_TO_SENTIMENT[i] for i in sorted(INDEX_TO_SENTIMENT)]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test set size: {len(all_labels)} samples")
    print(f"Overall Accuracy: {report['accuracy']:.2%}")
    print("\nPer-class metrics:")
    for cls in target_names:
        print(f"  {cls.upper():10s} - P: {report[cls]['precision']:.2%}  R: {report[cls]['recall']:.2%}  F1: {report[cls]['f1-score']:.2%}")
    print("="*60 + "\n")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(conf_matrix, target_names, args.output_dir / "confusion_matrix.png")
    
    # 2. Class Distribution
    plot_class_distribution(dataset.dataframe, args.output_dir / "class_distribution.png")
    
    # 3. Metrics by Class
    plot_metrics_by_class(report, args.output_dir / "metrics_by_class.png")
    
    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
