from torchvision import models
import torch.nn as nn

from src.dataset import get_dataloaders


def main():
    train_loader, _, _ = get_dataloaders()

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)

    images, labels = next(iter(train_loader))
    outputs = model(images)

    print("Input batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    main()