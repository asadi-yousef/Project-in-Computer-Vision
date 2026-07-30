"""Batched frozen-encoder feature extraction for a single dataset split.

Run once per (dataset, encoder, split); the resulting features are cached
(see cache.py) so no classifier training or evaluation ever re-runs the
image encoder, per stage_1.pdf's caching requirement.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def extract_features(
    encoder: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen encoder once over an entire dataset split.

    The encoder's own `.preprocess` must already be set as `dataset`'s
    image transform, so every batch arrives here already preprocessed.

    Args:
        encoder: a frozen encoder in eval mode (e.g. ResNet18Encoder).
        dataset: a dataset split whose `__getitem__` returns
            (preprocessed_image_tensor, label).
        device: device to run inference on.
        batch_size: inference batch size.
        num_workers: DataLoader worker count.

    Returns:
        (features, labels): tensors of shape (N, feature_dim) and (N,), in
        the dataset's own iteration order (no shuffling), so they line up
        index-for-index with `dataset._labels`.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    encoder = encoder.to(device)
    encoder.eval()

    feature_batches = []
    label_batches = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = encoder(images)
            feature_batches.append(features.cpu())
            label_batches.append(labels)

    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)
