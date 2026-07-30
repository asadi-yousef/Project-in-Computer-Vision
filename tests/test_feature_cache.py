import torch
import pytest

from src.features.cache import (
    FeatureCacheMetadata,
    cache_file_path,
    load_feature_cache_raw,
    save_feature_cache,
)


def test_cache_file_path_layout():
    path = cache_file_path("cache", "dtd", "resnet18", "train")
    assert str(path).replace("\\", "/") == "cache/dtd/resnet18/train.pt"


def test_save_and_load_round_trip(tmp_path):
    features = torch.randn(5, 8)
    labels = torch.tensor([0, 1, 2, 0, 1])
    metadata = FeatureCacheMetadata(
        dataset="dtd",
        encoder="resnet18",
        split="train",
        num_samples=5,
        feature_dim=8,
        num_classes=3,
    )
    path = tmp_path / "train.pt"

    save_feature_cache(path, features, labels, metadata)
    loaded = load_feature_cache_raw(path)

    assert torch.equal(loaded["features"], features)
    assert torch.equal(loaded["labels"], labels)
    assert loaded["metadata"]["dataset"] == "dtd"
    assert loaded["metadata"]["encoder"] == "resnet18"
    assert loaded["metadata"]["feature_dim"] == 8
    assert loaded["metadata"]["num_classes"] == 3


def test_load_missing_cache_raises_informative_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_features"):
        load_feature_cache_raw(tmp_path / "missing.pt")
