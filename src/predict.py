"""Live demo for sentiment prediction on single images."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

try:
    from .model import build_resnet18, get_device
    from .utils import load_config
    from .dataset import INDEX_TO_SENTIMENT
except ImportError:
    from model import build_resnet18, get_device
    from utils import load_config
    from dataset import INDEX_TO_SENTIMENT


def predict_single_image(image_path: Path | str, model, device, transform, cfg):
    """Predict sentiment for a single image."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error: Could not open image: {e}")
        return None
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(dim=1).item()
        confidence = probabilities[0, pred_idx].item()
    
    predicted_sentiment = INDEX_TO_SENTIMENT[pred_idx]
    
    return {
        "sentiment": predicted_sentiment,
        "confidence": confidence,
        "probabilities": {
            INDEX_TO_SENTIMENT[i]: probabilities[0, i].item()
            for i in range(len(INDEX_TO_SENTIMENT))
        },
        "image": image,
    }


def display_prediction(prediction, image_path):
    """Display image and prediction results."""
    if prediction is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Image
    axes[0].imshow(prediction["image"])
    axes[0].set_title(f"Image: {Path(image_path).name}", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Prediction results
    axes[1].axis("off")
    sentiment = prediction["sentiment"].upper()
    confidence = prediction["confidence"]
    
    # Main prediction
    result_text = f"PREDICTED SENTIMENT: {sentiment}\nCONFIDENCE: {confidence:.1%}"
    axes[1].text(0.5, 0.7, result_text, ha='center', va='center', fontsize=16, 
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # All probabilities
    prob_text = "Confidence Scores:\n\n"
    color_map = {"negative": "red", "neutral": "gray", "positive": "green"}
    for sentiment_name, prob in sorted(prediction["probabilities"].items(), key=lambda x: x[1], reverse=True):
        color = color_map[sentiment_name]
        bar = "█" * int(prob * 50)
        prob_text += f"{sentiment_name.capitalize():10s} {prob:6.1%} {bar}\n"
    
    axes[1].text(0.05, 0.35, prob_text, ha='left', va='top', fontsize=11, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live demo - predict sentiment for an image")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to config")
    parser.add_argument("--checkpoint", type=Path, default=Path("results/saved_models/best.pth"), help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Load model
    device = get_device()
    model = build_resnet18(num_classes=cfg.model.num_classes, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])
    
    # Predict
    print(f"\n🔍 Analyzing: {args.image_path}")
    prediction = predict_single_image(args.image_path, model, device, transform, cfg)
    
    if prediction:
        print(f"✅ Predicted: {prediction['sentiment'].upper()}")
        print(f"📊 Confidence: {prediction['confidence']:.1%}")
        print("\nDetailed probabilities:")
        for sentiment, prob in sorted(prediction["probabilities"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            print(f"  {sentiment.capitalize():10s} {prob:6.1%} {bar}")
        
        display_prediction(prediction, args.image_path)


if __name__ == "__main__":
    main()
