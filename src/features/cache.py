"""Feature cache file format.

Each (dataset, encoder, split) triple is cached as a single `.pt` file
holding its features, labels, and enough metadata to later tell whether the
cache still matches what a caller expects (Task 7 builds validation logic
on top of the raw loader defined here).
"""

import dataclasses
from pathlib import Path
from typing import Union

import torch


@dataclasses.dataclass
class FeatureCacheMetadata:
    """Describes what a cached feature file contains, for later validation."""

    dataset: str
    encoder: str
    split: str
    num_samples: int
    feature_dim: int
    num_classes: int


def cache_file_path(cache_dir: Union[str, Path], dataset: str, encoder: str, split: str) -> Path:
    """Where a given (dataset, encoder, split) feature cache lives on disk."""
    return Path(cache_dir) / dataset / encoder / f"{split}.pt"


def save_feature_cache(
    path: Union[str, Path],
    features: torch.Tensor,
    labels: torch.Tensor,
    metadata: FeatureCacheMetadata,
) -> None:
    """Save extracted features, labels, and metadata to a single cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "metadata": dataclasses.asdict(metadata),
        },
        path,
    )


def load_feature_cache_raw(path: Union[str, Path]) -> dict:
    """Load the raw cache dict (features, labels, metadata) from disk.

    This performs only file I/O; Task 7 wraps it with validation against an
    expected configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature cache found at {path}. Run scripts/extract_features.py first."
        )
    # weights_only=False: this cache is our own trusted local output, and its
    # metadata dict of plain str/int values doesn't need weights_only's
    # restricted unpickling.
    return torch.load(path, weights_only=False)
