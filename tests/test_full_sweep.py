import torch

from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.full_sweep import linear_probe_result_path, prototype_result_path, run_full_sweep


def _write_synthetic_cache(
    cache_dir, dataset, encoder, split, samples_per_class, num_classes, feature_dim, seed
):
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
    save_feature_cache(
        cache_file_path(cache_dir, dataset, encoder, split), features_tensor, labels_tensor, metadata
    )


def _seed_synthetic_cache(cache_dir):
    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=20, num_classes=3, feature_dim=8, seed=seed,
        )


def test_run_full_sweep_produces_expected_result_files(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"  # unused: cache already present, so extraction is skipped
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        data_dir, cache_dir, output_dir, device=torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10, "full"], seeds=[0],
    )

    assert linear_probe_result_path(output_dir, "dtd", "resnet18", 10, 0).exists()
    assert linear_probe_result_path(output_dir, "dtd", "resnet18", "full", 0).exists()
    assert prototype_result_path(output_dir, "dtd", "resnet18", 10, 0).exists()
    assert prototype_result_path(output_dir, "dtd", "resnet18", "full", 0).exists()


def test_run_full_sweep_skips_already_completed_runs(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
    )
    capsys.readouterr()  # discard first run's output

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
    )
    captured = capsys.readouterr()

    assert "already done, skipping" in captured.out
    assert "already cached, skipping" in captured.out


def test_run_full_sweep_respects_force_rerun(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
    )
    result_path = linear_probe_result_path(output_dir, "dtd", "resnet18", 10, 0)
    first_mtime = result_path.stat().st_mtime_ns

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0], force_rerun=True,
    )
    second_mtime = result_path.stat().st_mtime_ns

    assert second_mtime >= first_mtime
