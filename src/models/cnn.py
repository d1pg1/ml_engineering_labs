import torch
import torch.nn as nn


class CifarCNN(nn.Module):
    """CNN for CIFAR-10 image classification.

    Three convolutional blocks with BatchNorm and ReLU, followed by a
    dropout-regularized fully-connected classifier.

    Input:  (N, 3, 32, 32)
    Output: (N, n_classes)
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels, 32×32 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: 32 → 64 channels, 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 → 128 channels, 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
