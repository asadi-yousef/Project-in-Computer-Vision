"""Frozen ImageNet-1K-pretrained ResNet-18 feature extractor.

stage_1.pdf requires the 512-dimensional representation immediately before
the final classification layer, using the standard torchvision pretrained
checkpoint and its associated preprocessing, with all parameters frozen.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

FEATURE_DIM = 512


class ResNet18Encoder(nn.Module):
    """Frozen torchvision ResNet-18 with the final fc layer removed."""

    def __init__(self) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        # Keep every layer up to (and including) global average pooling;
        # drop only the final fc classification layer.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.preprocess = weights.transforms()

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "ResNet18Encoder":
        # This encoder must stay frozen and in eval mode no matter what a
        # caller does, since accidentally re-enabling training here would
        # silently violate the "frozen encoder" requirement.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim features from a preprocessed image batch.

        Args:
            images: batch tensor already run through `self.preprocess`.

        Returns:
            Tensor of shape (batch, 512).
        """
        features = self.backbone(images)
        return torch.flatten(features, start_dim=1)
