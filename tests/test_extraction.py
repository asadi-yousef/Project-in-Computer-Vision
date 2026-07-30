from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, Subset

from src.data.datasets import load_dtd_split
from src.encoders.resnet18 import FEATURE_DIM as RESNET18_FEATURE_DIM
from src.encoders.resnet18 import ResNet18Encoder
from src.features.extraction import extract_features


class _SyntheticDataset(Dataset):
    def __init__(self, n: int, dim: int):
        self.data = torch.arange(n * dim, dtype=torch.float32).reshape(n, dim)
        self.targets = torch.arange(n) % 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


class _IdentityEncoder(torch.nn.Module):
    def forward(self, x):
        return x


class _LinearEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_extract_features_preserves_order_and_shape():
    dataset = _SyntheticDataset(n=10, dim=4)
    features, labels = extract_features(
        _IdentityEncoder(), dataset, device=torch.device("cpu"), batch_size=3
    )

    assert features.shape == (10, 4)
    assert labels.shape == (10,)
    assert torch.equal(features, dataset.data)
    assert torch.equal(labels, dataset.targets)


def test_extract_features_output_has_no_gradient_tracking():
    dataset = _SyntheticDataset(n=4, dim=4)
    features, _ = extract_features(
        _LinearEncoder(), dataset, device=torch.device("cpu"), batch_size=2
    )
    assert not features.requires_grad


def test_extract_features_handles_batch_size_not_dividing_dataset_evenly():
    dataset = _SyntheticDataset(n=7, dim=4)
    features, labels = extract_features(
        _IdentityEncoder(), dataset, device=torch.device("cpu"), batch_size=3
    )
    assert features.shape == (7, 4)
    assert labels.shape == (7,)


# --- Integration check against the real, already-downloaded DTD dataset ---

_DTD_DOWNLOADED = (Path("data") / "dtd").exists()


@pytest.mark.skipif(not _DTD_DOWNLOADED, reason="DTD dataset not downloaded in ./data")
def test_extraction_end_to_end_on_a_few_real_dtd_images():
    encoder = ResNet18Encoder()
    train_split = load_dtd_split("data", split="train", download=False, transform=encoder.preprocess)
    tiny_subset = Subset(train_split, indices=list(range(4)))

    features, labels = extract_features(encoder, tiny_subset, device=torch.device("cpu"), batch_size=2)

    assert features.shape == (4, RESNET18_FEATURE_DIM)
    assert labels.shape == (4,)
    assert not features.requires_grad
