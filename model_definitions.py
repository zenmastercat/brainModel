import torch
import torch.nn as nn
from torchvision import models

# This file contains the exact model architectures used during training.
# It's crucial that these match the saved model weights.

class SimpleCNN(nn.Module):
    """
    Classification Model Definition.
    Matches the architecture used for training the classification task.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # Note: When loading a quantized model, the architecture is slightly different.
        # However, for loading the state_dict, we start with the original architecture.
        # The quantization happens dynamically in the server code.
        self.model = models.resnet18(weights=None) # We load our own weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class SimpleUNet(nn.Module):
    """
    Segmentation Model Definition.
    Matches the architecture used for training the segmentation task.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
