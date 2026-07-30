"""Frozen DINOv2 ViT-S/14 feature extractor (final class-token representation).

There is no torchvision implementation of DINOv2, so the official checkpoint
is loaded via `torch.hub` from Meta's `facebookresearch/dinov2` repository,
per stage_1.pdf's "publicly available pretrained checkpoint" requirement.
Used only on DTD in this project (see src/utils/config.py).
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

FEATURE_DIM = 384  # ViT-S/14 embedding dimension

# Official DINOv2 preprocessing: resize/crop to a multiple of the 14x14
# patch size, then normalize with standard ImageNet statistics.
_IMAGE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dinov2_preprocess() -> T.Compose:
    """Preprocessing pipeline matching the official DINOv2 checkpoint."""
    return T.Compose(
        [
            T.Resize(_IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(_IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 backbone, returning the final CLS-token embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.preprocess = build_dinov2_preprocess()

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "DINOv2Encoder":
        # See ResNet18Encoder.train: this encoder must never leave eval mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 384-dim CLS-token features from a preprocessed image batch.

        Args:
            images: batch tensor already run through `self.preprocess`.

        Returns:
            Tensor of shape (batch, 384).
        """
        return self.backbone(images)
