"""End-to-end feature-extraction pipeline: build an encoder, load official
splits with its preprocessing, extract, and cache. Shared by
scripts/extract_features.py and the full-sweep orchestration (src/full_sweep.py).
"""

from pathlib import Path
from typing import Union

import torch

from src.data.datasets import VALID_SPLITS, get_class_names, load_dataset_splits
from src.encoders.dinov2 import DINOv2Encoder
from src.encoders.resnet18 import ResNet18Encoder
from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.features.extraction import extract_features

ENCODER_BUILDERS = {
    "resnet18": ResNet18Encoder,
    "dinov2_vits14": DINOv2Encoder,
}


def all_splits_cached(cache_dir: Union[str, Path], dataset: str, encoder: str) -> bool:
    """Check whether every official split is already cached for (dataset, encoder)."""
    return all(
        cache_file_path(cache_dir, dataset, encoder, split).exists() for split in VALID_SPLITS
    )


def extract_and_cache_all_splits(
    dataset: str,
    encoder_name: str,
    data_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    device: torch.device,
    batch_size: int = 64,
) -> None:
    """Extract and cache every official split's features for one (dataset, encoder) pair."""
    if encoder_name == "dinov2_vits14" and dataset != "dtd":
        raise ValueError("dinov2_vits14 is only used on 'dtd' in this project")

    encoder = ENCODER_BUILDERS[encoder_name]()
    splits = load_dataset_splits(dataset, data_dir, download=False, transform=encoder.preprocess)
    num_classes = len(get_class_names(splits["train"]))

    for split_name in VALID_SPLITS:
        print(
            f"Extracting {dataset}/{encoder_name}/{split_name} ({len(splits[split_name])} images) ..."
        )
        features, labels = extract_features(
            encoder, splits[split_name], device=device, batch_size=batch_size
        )

        metadata = FeatureCacheMetadata(
            dataset=dataset,
            encoder=encoder_name,
            split=split_name,
            num_samples=features.shape[0],
            feature_dim=features.shape[1],
            num_classes=num_classes,
        )
        path = cache_file_path(cache_dir, dataset, encoder_name, split_name)
        save_feature_cache(path, features, labels, metadata)
        print(f"  saved {features.shape[0]} x {features.shape[1]} features to {path}")
