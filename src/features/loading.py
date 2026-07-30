"""Load cached features with validation against what a caller expects.

Task 6 produces cache files; this is the "read side" that classifier
training/evaluation uses. It fails loudly if a cache is stale or was built
for the wrong dataset/encoder/split, rather than silently training on
mismatched features.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import torch

from src.features.cache import cache_file_path, load_feature_cache_raw

VALID_SPLITS = ("train", "val", "test")


def load_validated_feature_cache(
    cache_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    split: str,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Load a cached (features, labels) pair, checking its metadata matches
    the (dataset, encoder, split) requested.

    Returns:
        (features, labels, metadata).

    Raises:
        ValueError: if the cache's stored metadata does not match the
            requested dataset/encoder/split, or its tensor shapes are
            inconsistent with its own metadata (a corrupted/stale cache).
    """
    path = cache_file_path(cache_dir, dataset, encoder, split)
    raw = load_feature_cache_raw(path)
    features, labels, metadata = raw["features"], raw["labels"], raw["metadata"]

    if metadata["dataset"] != dataset:
        raise ValueError(
            f"Cache at {path} was built for dataset {metadata['dataset']!r}, expected {dataset!r}"
        )
    if metadata["encoder"] != encoder:
        raise ValueError(
            f"Cache at {path} was built with encoder {metadata['encoder']!r}, expected {encoder!r}"
        )
    if metadata["split"] != split:
        raise ValueError(f"Cache at {path} is for split {metadata['split']!r}, expected {split!r}")
    if features.shape[0] != metadata["num_samples"]:
        raise ValueError(
            f"Cache at {path} metadata says {metadata['num_samples']} samples "
            f"but features tensor has {features.shape[0]}"
        )
    if features.shape[1] != metadata["feature_dim"]:
        raise ValueError(
            f"Cache at {path} metadata says feature_dim={metadata['feature_dim']} "
            f"but features tensor has dim {features.shape[1]}"
        )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"Cache at {path} has {features.shape[0]} features but {labels.shape[0]} labels"
        )

    return features, labels, metadata


def load_all_splits(
    cache_dir: Union[str, Path], dataset: str, encoder: str
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, dict]]:
    """Load and cross-validate train/val/test caches for one (dataset, encoder) pair.

    Beyond each split's own validation, this checks that feature_dim and
    num_classes agree across all three splits, since a mismatch there would
    mean the caches came from two different encoder runs.

    Raises:
        ValueError: if feature_dim or num_classes disagree across splits.
    """
    splits = {
        split: load_validated_feature_cache(cache_dir, dataset, encoder, split)
        for split in VALID_SPLITS
    }

    feature_dims = {split: meta["feature_dim"] for split, (_, _, meta) in splits.items()}
    if len(set(feature_dims.values())) > 1:
        raise ValueError(
            f"Inconsistent feature_dim across splits for {dataset}/{encoder}: {feature_dims}"
        )

    num_classes_by_split = {split: meta["num_classes"] for split, (_, _, meta) in splits.items()}
    if len(set(num_classes_by_split.values())) > 1:
        raise ValueError(
            f"Inconsistent num_classes across splits for {dataset}/{encoder}: {num_classes_by_split}"
        )

    return splits
