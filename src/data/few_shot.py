"""Balanced few-shot subset sampling from the official training split.

stage_1.pdf requires K in {5, 10} images per class, sampled from the
official training split only, with deterministic seeds {0, 1, 2}. This
module implements exactly that sampling, independent of which dataset or
encoder is used downstream.
"""

import random
from collections import defaultdict
from typing import List

from torch.utils.data import Dataset, Subset


def get_labels(dataset: Dataset) -> List[int]:
    """Return the integer label of every sample in `dataset`, in order.

    Reads the dataset's own `_labels` list rather than iterating
    `dataset[i]`, since the latter would decode every image just to read
    its label.
    """
    if not hasattr(dataset, "_labels"):
        raise AttributeError(
            f"{type(dataset).__name__} has no `_labels` attribute; "
            "this helper only supports the DTD and Flowers102 torchvision datasets."
        )
    return list(dataset._labels)


def sample_balanced_subset_indices(
    labels: List[int], k_per_class: int, seed: int
) -> List[int]:
    """Deterministically choose exactly `k_per_class` indices for every class.

    Args:
        labels: label of every sample in the source dataset, in order.
        k_per_class: number of samples to draw per class (5 or 10 here).
        seed: controls which samples are drawn; the same seed and `labels`
            always produce the same indices.

    Returns:
        A sorted list of dataset indices containing exactly `k_per_class`
        entries for each distinct label in `labels`.

    Raises:
        ValueError: if any class has fewer than `k_per_class` samples.
    """
    indices_by_class = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_class[label].append(index)

    rng = random.Random(seed)
    selected_indices = []
    for label in sorted(indices_by_class):
        class_indices = indices_by_class[label]
        if len(class_indices) < k_per_class:
            raise ValueError(
                f"Class {label} has only {len(class_indices)} training images, "
                f"fewer than the requested k_per_class={k_per_class}"
            )
        selected_indices.extend(rng.sample(class_indices, k_per_class))

    return sorted(selected_indices)


def make_few_shot_subset(dataset: Dataset, k_per_class: int, seed: int) -> Subset:
    """Build a class-balanced k-shot `Subset` of the official training split.

    Args:
        dataset: the official training-split dataset (never val/test).
        k_per_class: 5 or 10, per stage_1.pdf.
        seed: one of {0, 1, 2}, per stage_1.pdf.

    Returns:
        A `torch.utils.data.Subset` containing exactly `k_per_class` images
        for each class.
    """
    labels = get_labels(dataset)
    indices = sample_balanced_subset_indices(labels, k_per_class, seed)
    return Subset(dataset, indices)
