import json
from pathlib import Path

import torch

from src.classifiers.linear_probe_runner import run_linear_probe_experiment
from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.utils.config import ExperimentConfig, LinearProbeHyperparams


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


def test_run_linear_probe_experiment_end_to_end_on_synthetic_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"

    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=20, num_classes=3, feature_dim=8, seed=seed,
        )

    config = ExperimentConfig(
        dataset="dtd", encoder="resnet18", method="linear_probe", k_shot=10, seed=0,
        linear_probe=LinearProbeHyperparams(learning_rate=1e-2, weight_decay=0.0, max_epochs=30, batch_size=8),
    )

    result = run_linear_probe_experiment(config, cache_dir, output_dir, device=torch.device("cpu"))

    assert result["test_accuracy"] > 0.5  # classes are well-separated by construction
    run_dir = Path(result["run_dir"])
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "checkpoint.pt").exists()

    with open(run_dir / "history.json") as f:
        history = json.load(f)
    assert len(history) == 30  # == max_epochs


def test_run_linear_probe_experiment_uses_only_k_shot_subset_of_training_data(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"

    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=50, num_classes=3, feature_dim=8, seed=seed,
        )

    config = ExperimentConfig(
        dataset="dtd", encoder="resnet18", method="linear_probe", k_shot=5, seed=1,
        linear_probe=LinearProbeHyperparams(max_epochs=5, batch_size=8),
    )

    result = run_linear_probe_experiment(config, cache_dir, output_dir, device=torch.device("cpu"))

    # 5-shot with 3 classes -> exactly 15 training images were used, not all 150.
    run_dir = Path(result["run_dir"])
    assert "k5" in str(run_dir)
    assert "seed1" in str(run_dir)
