import torch
import pytest

from src.features.cache import FeatureCacheMetadata, save_feature_cache, cache_file_path
from src.features.loading import load_all_splits, load_validated_feature_cache


def _write_cache(cache_dir, dataset, encoder, split, num_samples=6, feature_dim=4, num_classes=3):
    features = torch.randn(num_samples, feature_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    metadata = FeatureCacheMetadata(
        dataset=dataset,
        encoder=encoder,
        split=split,
        num_samples=num_samples,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
    path = cache_file_path(cache_dir, dataset, encoder, split)
    save_feature_cache(path, features, labels, metadata)
    return features, labels


def test_load_validated_feature_cache_returns_matching_data(tmp_path):
    features, labels = _write_cache(tmp_path, "dtd", "resnet18", "train")
    loaded_features, loaded_labels, metadata = load_validated_feature_cache(
        tmp_path, "dtd", "resnet18", "train"
    )
    assert torch.equal(loaded_features, features)
    assert torch.equal(loaded_labels, labels)
    assert metadata["dataset"] == "dtd"


def test_dataset_mismatch_between_path_and_metadata_raises(tmp_path):
    # Simulates a cache file living at the "dtd" path but whose stored
    # metadata says it was built for a different dataset (e.g. a stale or
    # misplaced cache file).
    features = torch.randn(6, 4)
    labels = torch.randint(0, 3, (6,))
    mismatched_metadata = FeatureCacheMetadata(
        dataset="flowers102", encoder="resnet18", split="train",
        num_samples=6, feature_dim=4, num_classes=3,
    )
    path = cache_file_path(tmp_path, "dtd", "resnet18", "train")
    save_feature_cache(path, features, labels, mismatched_metadata)

    with pytest.raises(ValueError, match="dataset"):
        load_validated_feature_cache(tmp_path, "dtd", "resnet18", "train")


def test_encoder_mismatch_between_path_and_metadata_raises(tmp_path):
    features = torch.randn(6, 4)
    labels = torch.randint(0, 3, (6,))
    mismatched_metadata = FeatureCacheMetadata(
        dataset="dtd", encoder="dinov2_vits14", split="train",
        num_samples=6, feature_dim=4, num_classes=3,
    )
    path = cache_file_path(tmp_path, "dtd", "resnet18", "train")
    save_feature_cache(path, features, labels, mismatched_metadata)

    with pytest.raises(ValueError, match="encoder"):
        load_validated_feature_cache(tmp_path, "dtd", "resnet18", "train")


def test_corrupted_num_samples_metadata_raises(tmp_path):
    features = torch.randn(6, 4)
    labels = torch.randint(0, 3, (6,))
    bad_metadata = FeatureCacheMetadata(
        dataset="dtd", encoder="resnet18", split="train",
        num_samples=999,  # deliberately wrong
        feature_dim=4, num_classes=3,
    )
    path = cache_file_path(tmp_path, "dtd", "resnet18", "train")
    save_feature_cache(path, features, labels, bad_metadata)

    with pytest.raises(ValueError, match="num_samples"):
        load_validated_feature_cache(tmp_path, "dtd", "resnet18", "train")


def test_load_all_splits_passes_for_consistent_caches(tmp_path):
    for split in ("train", "val", "test"):
        _write_cache(tmp_path, "dtd", "resnet18", split, feature_dim=4, num_classes=3)

    splits = load_all_splits(tmp_path, "dtd", "resnet18")
    assert set(splits.keys()) == {"train", "val", "test"}


def test_load_all_splits_detects_inconsistent_feature_dim(tmp_path):
    _write_cache(tmp_path, "dtd", "resnet18", "train", feature_dim=4, num_classes=3)
    _write_cache(tmp_path, "dtd", "resnet18", "val", feature_dim=8, num_classes=3)  # mismatched dim
    _write_cache(tmp_path, "dtd", "resnet18", "test", feature_dim=4, num_classes=3)

    with pytest.raises(ValueError, match="feature_dim"):
        load_all_splits(tmp_path, "dtd", "resnet18")
