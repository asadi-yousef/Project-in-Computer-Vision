from pathlib import Path

import pytest
import torch

from src.classifiers.linear_probe import LinearProbe
from src.evaluation.predictions import (
    get_linear_probe_test_predictions,
    get_prototype_test_predictions,
)
from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache


def _write_synthetic_cache(cache_dir, dataset, encoder, split, samples_per_class, num_classes, feature_dim, seed):
    generator = torch.Generator().manual_seed(seed)
    features, labels = [], []
    for class_index in range(num_classes):
        center = torch.zeros(feature_dim)
        center[class_index % feature_dim] = 8.0
        noise = torch.randn(samples_per_class, feature_dim, generator=generator) * 0.1
        features.append(center.unsqueeze(0) + noise)
        labels.append(torch.full((samples_per_class,), class_index, dtype=torch.long))
    features_tensor, labels_tensor = torch.cat(features), torch.cat(labels)
    metadata = FeatureCacheMetadata(
        dataset=dataset, encoder=encoder, split=split,
        num_samples=features_tensor.shape[0], feature_dim=feature_dim, num_classes=num_classes,
    )
    save_feature_cache(cache_file_path(cache_dir, dataset, encoder, split), features_tensor, labels_tensor, metadata)
    return features_tensor, labels_tensor


def test_get_linear_probe_test_predictions_matches_a_perfect_checkpoint(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"

    _write_synthetic_cache(cache_dir, "dtd", "resnet18", "test", samples_per_class=5, num_classes=3, feature_dim=8, seed=0)

    # A hand-built linear probe whose weight IS the one-hot class-center
    # direction, so it perfectly separates these synthetic features.
    model = LinearProbe(feature_dim=8, num_classes=3)
    with torch.no_grad():
        model.linear.weight.zero_()
        for class_index in range(3):
            model.linear.weight[class_index, class_index] = 1.0
        model.linear.bias.zero_()

    checkpoint_dir = output_dir / "linear_probe" / "dtd" / "resnet18" / "k10" / "seed0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "checkpoint.pt")

    true_labels, predicted_labels = get_linear_probe_test_predictions(
        cache_dir, output_dir, "dtd", "resnet18", k_shot=10, seed=0, device=torch.device("cpu"),
    )

    assert torch.equal(true_labels, predicted_labels)


def test_get_linear_probe_test_predictions_raises_for_missing_checkpoint(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _write_synthetic_cache(cache_dir, "dtd", "resnet18", "test", samples_per_class=5, num_classes=3, feature_dim=8, seed=0)

    with pytest.raises(FileNotFoundError, match="run_linear_probe"):
        get_linear_probe_test_predictions(
            cache_dir, output_dir, "dtd", "resnet18", k_shot=10, seed=0, device=torch.device("cpu"),
        )


def test_get_prototype_test_predictions_is_accurate_on_well_separated_synthetic_data(tmp_path):
    cache_dir = tmp_path / "cache"

    _write_synthetic_cache(cache_dir, "dtd", "resnet18", "train", samples_per_class=20, num_classes=3, feature_dim=8, seed=0)
    _write_synthetic_cache(cache_dir, "dtd", "resnet18", "test", samples_per_class=10, num_classes=3, feature_dim=8, seed=1)

    true_labels, predicted_labels = get_prototype_test_predictions(
        cache_dir, "dtd", "resnet18", k_shot="full", seed=0,
    )

    accuracy = (true_labels == predicted_labels).float().mean().item()
    assert accuracy > 0.9
