from pathlib import Path
from typing import Tuple, Dict

import yaml
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def load_config(config_path: str = "configs/base.yaml") -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Relative path to config file from project root.

    Returns:
        Parsed config dictionary.
    """
    project_root = Path(__file__).resolve().parents[1]
    full_config_path = project_root / config_path

    if not full_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_config_path}")

    with open(full_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {full_config_path}")

    return config


def validate_config(config: Dict) -> None:
    """
    Validate presence of required config keys.
    """
    required_paths = [
        ("data", "annotations"),
        ("data", "raw_root"),
        ("data", "processed_root"),
        ("data", "mean"),
        ("data", "std"),
        ("training", "seed"),
        ("training", "batch_size"),
        ("training", "val_split"),
        ("training", "test_split"),
    ]

    for section, key in required_paths:
        if section not in config or key not in config[section]:
            raise KeyError(f"Missing config key: {section}.{key}")

    val_split = config["training"]["val_split"]
    test_split = config["training"]["test_split"]

    if val_split <= 0 or test_split <= 0:
        raise ValueError("val_split and test_split must be greater than 0.")

    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be less than 1.")


def load_annotations(config: Dict) -> pd.DataFrame:
    """
    Load and validate annotation CSV, map labels, and build full image paths.

    Returns:
        DataFrame with columns:
        - image_path
        - sentiment
        - label
        - full_image_path
    """
    project_root = Path(__file__).resolve().parents[1]
    annotations_path = project_root / config["data"]["annotations"]

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    df = pd.read_csv(annotations_path)

    required_columns = {"image_path", "sentiment"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in annotations CSV: {missing_columns}")

    df = df[["image_path", "sentiment"]].copy()

    # Clean string columns
    df["image_path"] = df["image_path"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Map sentiment labels
    df["label"] = df["sentiment"].map(LABEL_MAP)

    invalid_labels = df[df["label"].isna()]["sentiment"].unique().tolist()
    if invalid_labels:
        raise ValueError(f"Invalid sentiment labels found: {invalid_labels}")

    raw_root = project_root / config["data"]["raw_root"]
    df["full_image_path"] = df["image_path"].apply(lambda x: raw_root / x)

    # Filter only existing image files
    missing_files = df[~df["full_image_path"].apply(lambda x: x.exists())]
    if not missing_files.empty:
        print(f"Warning: {len(missing_files)} image files are missing and will be removed.")

    df = df[df["full_image_path"].apply(lambda x: x.exists())].copy()

    if df.empty:
        raise ValueError("No valid image files found after filtering missing paths.")

    df.reset_index(drop=True, inplace=True)
    return df


def split_data(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test with stratification.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=val_split + test_split,
        random_state=seed,
        stratify=df["label"],
    )

    relative_test_size = test_split / (val_split + test_split)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> None:
    """
    Save split CSVs for reproducibility.
    """
    project_root = Path(__file__).resolve().parents[1]
    processed_root = project_root / config["data"]["processed_root"]
    processed_root.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_root / "train_split.csv", index=False)
    val_df.to_csv(processed_root / "val_split.csv", index=False)
    test_df.to_csv(processed_root / "test_split.csv", index=False)


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Print sizes and label distribution for each split.
    """
    def summarize_split(name: str, split_df: pd.DataFrame) -> None:
        print(f"\n{name} split size: {len(split_df)}")
        counts = split_df["sentiment"].value_counts().sort_index()
        percentages = split_df["sentiment"].value_counts(normalize=True).sort_index() * 100

        print("Class distribution:")
        for label in counts.index:
            print(f"  {label}: {counts[label]} ({percentages[label]:.2f}%)")

    print("\n===== DATA SPLIT SUMMARY =====")
    summarize_split("Train", train_df)
    summarize_split("Validation", val_df)
    summarize_split("Test", test_df)


def get_transforms(config: Dict):
    """
    Create training and evaluation transforms from config.
    """
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    train_transforms = [transforms.Resize((224, 224))]
    if config.get("augmentation", {}).get("horizontal_flip", False):
        flip_prob = config.get("augmentation", {}).get("horizontal_flip_prob", 0.5)
        train_transforms.append(transforms.RandomHorizontalFlip(p=flip_prob))

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(eval_transforms)


class SentimentImageDataset(Dataset):
    """
    PyTorch Dataset for image-based sentiment classification.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        image_path = row["full_image_path"]
        label = int(row["label"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {image_path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(config_path: str = "configs/base.yaml"):
    """
    Main entry point for the rest of the team.

    Returns:
        train_loader, val_loader, test_loader
    """
    config = load_config(config_path)
    validate_config(config)

    df = load_annotations(config)

    train_df, val_df, test_df = split_data(
        df=df,
        val_split=config["training"]["val_split"],
        test_split=config["training"]["test_split"],
        seed=config["training"]["seed"],
    )

    if config.get("debug", {}).get("save_splits", False):
        save_split_csvs(train_df, val_df, test_df, config)

    if config.get("debug", {}).get("print_split_summary", False):
        print_split_summary(train_df, val_df, test_df)

    train_transform, eval_transform = get_transforms(config)

    train_dataset = SentimentImageDataset(train_df, transform=train_transform)
    val_dataset = SentimentImageDataset(val_df, transform=eval_transform)
    test_dataset = SentimentImageDataset(test_df, transform=eval_transform)

    batch_size = config["training"]["batch_size"]
    num_workers = config.get("dataloader", {}).get("num_workers", 0)
    pin_memory = config.get("dataloader", {}).get("pin_memory", False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def main():
    """
    Quick smoke test for dataset pipeline.
    """
    train_loader, val_loader, test_loader = get_dataloaders()

    print("\n===== DATALOADER CHECK =====")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Sample labels: {labels[:10]}")


if __name__ == "__main__":
    main()