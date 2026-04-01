# ??? Image-Based Sentiment Analysis

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg) ![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white) ![Status](https://img.shields.io/badge/status-research-blueviolet) ![License](https://img.shields.io/badge/license-academic-lightgrey)

Image-based sentiment analysis aims to infer affect (positive, neutral, negative) purely from visual cues. This repository investigates how far we can push sentiment understanding without any textual context by fine-tuning a ResNet-18 backbone on a curated set of social-media imagery.

> **TL;DR**: Transfer-learn ResNet-18 ? classify visual sentiment ? analyze when pictures alone succeed or fail.

---

## Table of Contents
1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Getting Started](#getting-started)
6. [Training & Evaluation Workflow](#training--evaluation-workflow)
7. [Methodology](#methodology)
8. [Results & Artifacts](#results--artifacts)
9. [Analysis & Limitations](#analysis--limitations)
10. [Roadmap](#roadmap)
11. [Team](#team)
12. [License](#license)

## Overview
- Focus: classify sentiment directly from images using deep convolutional networks.
- Backbone: ResNet-18 with ImageNet-pretrained weights for rapid convergence.
- Deliverables: trained models, quantitative metrics, qualitative error analysis, and a written report summarizing findings.

## Objectives
- Build an end-to-end pipeline for three-way sentiment classification (positive/neutral/negative).
- Benchmark performance with accuracy, precision, recall, F1, and confusion matrices.
- Investigate failure cases to understand when visual input is insufficient.
- Document assumptions, limitations, and opportunities for multimodal extensions.

## Project Structure
`	ext
image-sentiment-analysis/
+-- data/
¦   +-- raw/             # place original images + labels here (not tracked)
¦   +-- processed/       # standardized tensors / metadata
+-- notebooks/           # exploratory analysis & prototyping
+-- src/
¦   +-- dataset.py       # PyTorch Dataset + transforms
¦   +-- model.py         # ResNet-18 head + utility blocks
¦   +-- train.py         # training loop, logging, checkpointing
¦   +-- evaluate.py      # metrics, confusion matrix, qualitative dumps
¦   +-- utils.py         # helpers (seeding, config parsing, plotting)
+-- results/
¦   +-- figures/         # training curves, Grad-CAM, confusion matrix
¦   +-- tables/          # metrics in CSV / markdown form
¦   +-- saved_models/    # best checkpoints (.pth)
+-- report/
¦   +-- report.md        # final technical write-up
+-- requirements.txt     # python dependencies
+-- README.md            # you are here
`

> **Note**: Only the high-level structure is versioned here. Populate folders as you run experiments.

## Dataset
- Source: curated social-media images labeled with coarse sentiment labels (positive / neutral / negative).
- Size: specify once finalized (e.g., ~10k images) along with label distribution.
- Storage: data/raw/ for untouched assets, data/processed/ for resized (224×224), normalized tensors.
- Privacy: ensure redistribution rights before committing raw data.

## Getting Started
1. **Clone & enter the repo**
   `ash
   git clone <your-repo-url>
   cd image-sentiment-analysis
   `
2. **Create a virtual environment**
   `ash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   `
3. **Install dependencies**
   `ash
   pip install -r requirements.txt
   `
4. **Add the dataset**
   `	ext
   data/
   +-- raw/
       +-- images/*.jpg
       +-- annotations.csv  # image_path,sentiment
   `
5. **(Optional) Configure paths/parameters**
   - Use environment variables or a YAML/JSON config (see src/utils.py once implemented).

## Training & Evaluation Workflow
`ash
# Train
python src/train.py \
  --config configs/base.yaml \
  --data_root data/processed \
  --save_dir results/saved_models

# Evaluate best checkpoint
python src/evaluate.py \
  --checkpoint results/saved_models/best.pth \
  --split test
`
- Training logs + tensorboard summaries are written to esults/.
- Evaluation script exports classification metrics and confusion matrices to esults/tables/ + esults/figures/.

## Methodology
1. **Data Preprocessing**
   - Resize/crop to 224×224, normalize with ImageNet stats, augment (horizontal flip, color jitter) for robustness.
   - Split dataset into train/val/test (e.g., 70/15/15) while preserving label balance.
2. **Model**
   - Load torchvision ResNet-18 pretrained on ImageNet.
   - Replace final FC layer with a 3-class head; optionally add dropout for regularization.
3. **Training**
   - Loss: CrossEntropyLoss.
   - Optimizer: Adam (default lr=3e-4) with cosine or step LR scheduler.
   - Early stopping based on validation F1; checkpoints saved in esults/saved_models/.
4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, Macro-F1.
   - Diagnostics: confusion matrix heatmap, per-class ROC, Grad-CAM visualizations (optional).

## Results & Artifacts
| Split  | Accuracy | Macro F1 | Notes |
|--------|----------|----------|-------|
| Train  | _TBD_    | _TBD_    | Fill after training |
| Val    | _TBD_    | _TBD_    | |
| Test   | _TBD_    | _TBD_    | |

- Plots live under esults/figures/ (loss curves, confusion matrices, interpretability visuals).
- Numerical summaries go to esults/tables/ as CSV + Markdown for direct inclusion in the report.
- Best-performing checkpoints are stored in esults/saved_models/.

## Analysis & Limitations
- **Error analysis**: Inspect misclassified samples to understand ambiguous expressions, occlusions, or cultural references.
- **Limitations**:
  - Visual sentiment can be underspecified without captions/text.
  - Labels may contain subjectivity or noise from annotation.
  - Domain transfer (e.g., memes vs. product photos) may degrade accuracy.
- **Opportunities**:
  - Add multimodal cues (image + caption) or CLIP-style embeddings.
  - Explore curriculum learning or contrastive pretraining to emphasize affective cues.

## Roadmap
- [ ] Finalize dataset curation + document statistics.
- [ ] Implement data augmentation + dataloader caching.
- [ ] Integrate experiment tracking (Weights & Biases or TensorBoard).
- [ ] Add Grad-CAM / attention rollout visualizations.
- [ ] Extend to multimodal inputs (image + short text).

## Team
- Isaac
- Selvi
- Alhajie
- Macha

## License
Academic use only. Please reach out to the team before reusing the dataset or trained weights.
