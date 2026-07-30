"""Orchestrates one full linear-probe experiment: load cached features,
restrict to a K-shot subset if applicable, train, evaluate on test exactly
once, and save every output the spec requires alongside the config.
"""

import dataclasses
import json
from pathlib import Path
from typing import Union

import torch

from src.classifiers.linear_probe import evaluate_linear_probe, train_linear_probe
from src.data.few_shot import sample_balanced_subset_indices
from src.features.loading import load_validated_feature_cache
from src.utils.config import ExperimentConfig, save_config
from src.utils.run_metadata import build_run_metadata, save_run_metadata


def run_linear_probe_experiment(
    config: ExperimentConfig,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
) -> dict:
    """Run one (dataset, encoder, k_shot, seed) linear-probe experiment end to end.

    Args:
        config: must have method="linear_probe".
        cache_dir: directory holding cached features (see src/features/cache.py).
        output_dir: directory to save this run's config/history/result/checkpoint under.
        device: device to train and evaluate on.

    Returns:
        A dict with test_accuracy, best_val_accuracy, best_epoch, and the
        run's output directory.
    """
    if config.method != "linear_probe":
        raise ValueError(f"config.method must be 'linear_probe', got {config.method!r}")

    train_features, train_labels, train_metadata = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "train"
    )
    val_features, val_labels, _ = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "val"
    )
    test_features, test_labels, _ = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "test"
    )
    num_classes = train_metadata["num_classes"]

    if config.k_shot != "full":
        indices = sample_balanced_subset_indices(train_labels.tolist(), config.k_shot, config.seed)
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    result = train_linear_probe(
        train_features,
        train_labels,
        val_features,
        val_labels,
        num_classes=num_classes,
        hyperparams=config.linear_probe,
        seed=config.seed,
        device=device,
    )

    test_accuracy = evaluate_linear_probe(
        result.best_state_dict,
        feature_dim=train_features.shape[1],
        num_classes=num_classes,
        test_features=test_features,
        test_labels=test_labels,
        device=device,
    )

    run_dir = Path(output_dir) / config.dataset / config.encoder / f"k{config.k_shot}" / f"seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config.yaml")

    with open(run_dir / "history.json", "w") as f:
        json.dump([dataclasses.asdict(epoch_log) for epoch_log in result.history], f, indent=2)

    metadata = build_run_metadata(config, device)
    metadata["result"] = {
        "test_accuracy": test_accuracy,
        "best_val_accuracy": result.best_val_accuracy,
        "best_epoch": result.best_epoch,
    }
    save_run_metadata(metadata, run_dir / "result.json")
    torch.save(result.best_state_dict, run_dir / "checkpoint.pt")

    return {
        "test_accuracy": test_accuracy,
        "best_val_accuracy": result.best_val_accuracy,
        "best_epoch": result.best_epoch,
        "run_dir": str(run_dir),
    }
