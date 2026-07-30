from pathlib import Path

import torch

from src.classifiers.prototype_runner import run_prototype_experiment
from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.utils.config import ExperimentConfig


def _write_synthetic_cache(
    cache_dir, dataset, encoder, split, samples_per_class, num_classes, feature_dim, seed
):
    generator = torch.Generator().manual_seed(seed)
    features = []
    labels = []
    for class_index in range(num_classes):
        center = torch.zeros(feature_dim)
        center[class_index % feature_dim] = 8.0
        noise = torch.randn(samples_per_class, feature_dim, generator=generator) * 0.1
        features.append(center.unsqueeze(0) + noise)
        labels.append(torch.full((samples_per_class,), class_index, dtype=torch.long))
    features_tensor = torch.cat(features)
    labels_tensor = torch.cat(labels)

    metadata = FeatureCacheMetadata(
        dataset=dataset,
        encoder=encoder,
        split=split,
        num_samples=features_tensor.shape[0],
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
    save_feature_cache(cache_file_path(cache_dir, dataset, encoder, split), features_tensor, labels_tensor, metadata)


def test_run_prototype_experiment_k_shot_end_to_end_on_synthetic_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"

    for split, seed in [("train", 0), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=20, num_classes=3, feature_dim=8, seed=seed,
        )

    config = ExperimentConfig(dataset="dtd", encoder="resnet18", method="prototype", k_shot=10, seed=0)
    result = run_prototype_experiment(config, cache_dir, output_dir)

    assert result["test_accuracy"] > 0.5
    run_dir = Path(result["run_dir"])
    assert "seed0" in str(run_dir)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "result.json").exists()


def test_run_prototype_experiment_full_setting_uses_single_run_folder(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"

    for split, seed in [("train", 0), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=20, num_classes=3, feature_dim=8, seed=seed,
        )

    config = ExperimentConfig(dataset="dtd", encoder="resnet18", method="prototype", k_shot="full", seed=0)
    result = run_prototype_experiment(config, cache_dir, output_dir)

    assert result["test_accuracy"] > 0.5
    run_dir = Path(result["run_dir"])
    assert "single_run" in str(run_dir)
