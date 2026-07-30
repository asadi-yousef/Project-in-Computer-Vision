"""Orchestrates one full image-prototype experiment: load cached features,
restrict to a K-shot subset if applicable, build prototypes, classify the
test split, and save the config/result alongside it.
"""

from pathlib import Path
from typing import Union

import torch

from src.classifiers.prototype import compute_class_prototypes, predict_by_cosine_similarity
from src.data.few_shot import sample_balanced_subset_indices
from src.features.loading import load_validated_feature_cache
from src.utils.config import ExperimentConfig, save_config
from src.utils.run_metadata import build_run_metadata, save_run_metadata


def run_prototype_experiment(
    config: ExperimentConfig,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> dict:
    """Run one (dataset, encoder, k_shot[, seed]) image-prototype experiment end to end.

    No training occurs: prototypes are computed directly from the (possibly
    K-shot-subsetted) cached training features, per stage_1.pdf Option A.

    Args:
        config: must have method="prototype". For k_shot in {5, 10},
            config.seed selects the balanced subset; for k_shot="full" the
            seed is unused (the full-data result requires only one run).
        cache_dir: directory holding cached features.
        output_dir: directory to save this run's config/result under.

    Returns:
        A dict with test_accuracy and the run's output directory.
    """
    if config.method != "prototype":
        raise ValueError(f"config.method must be 'prototype', got {config.method!r}")

    train_features, train_labels, train_metadata = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "train"
    )
    test_features, test_labels, _ = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "test"
    )
    num_classes = train_metadata["num_classes"]

    if config.k_shot != "full":
        indices = sample_balanced_subset_indices(train_labels.tolist(), config.k_shot, config.seed)
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    prototypes = compute_class_prototypes(train_features, train_labels, num_classes)
    predictions = predict_by_cosine_similarity(test_features, prototypes)
    test_accuracy = (predictions == test_labels).float().mean().item()

    seed_folder = f"seed{config.seed}" if config.k_shot != "full" else "single_run"
    run_dir = Path(output_dir) / config.dataset / config.encoder / f"k{config.k_shot}" / seed_folder
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config.yaml")

    metadata = build_run_metadata(config, torch.device("cpu"))
    metadata["result"] = {"test_accuracy": test_accuracy}
    save_run_metadata(metadata, run_dir / "result.json")

    return {"test_accuracy": test_accuracy, "run_dir": str(run_dir)}
