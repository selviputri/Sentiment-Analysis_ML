import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_FILE = PROJECT_ROOT / "data" / "raw" / "MVSA_Single" / "labelResultAll.txt"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "annotations.csv"
IMAGE_DIR = "MVSA_Single/data"

df = pd.read_csv(LABEL_FILE, sep="\t", header=0, names=["id", "labels"])
df["image_label"] = df["labels"].str.split(",").str[1].str.strip()
df["image_path"] = df["id"].apply(lambda x: f"{IMAGE_DIR}/{x}.jpg")
df = df[["image_path", "image_label"]].rename(columns={"image_label": "sentiment"})
df = df[df["sentiment"].isin(["positive", "neutral", "negative"])].reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_CSV}")