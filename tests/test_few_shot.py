from collections import Counter
from pathlib import Path

import pytest

from src.data.datasets import load_dtd_split
from src.data.few_shot import (
    get_labels,
    make_few_shot_subset,
    sample_balanced_subset_indices,
)


class _FakeDatasetWithLabels:
    def __init__(self, labels):
        self._labels = labels

    def __len__(self):
        return len(self._labels)


def _synthetic_labels(num_classes: int, images_per_class: int):
    # e.g. num_classes=3, images_per_class=4 -> [0,0,0,0,1,1,1,1,2,2,2,2]
    return [c for c in range(num_classes) for _ in range(images_per_class)]


def test_get_labels_reads_internal_attribute():
    dataset = _FakeDatasetWithLabels([0, 1, 0, 1])
    assert get_labels(dataset) == [0, 1, 0, 1]


def test_get_labels_raises_informative_error_when_unsupported():
    class NoLabels:
        pass

    with pytest.raises(AttributeError, match="_labels"):
        get_labels(NoLabels())


def test_sample_balanced_subset_selects_exactly_k_per_class():
    labels = _synthetic_labels(num_classes=5, images_per_class=20)
    indices = sample_balanced_subset_indices(labels, k_per_class=5, seed=0)

    selected_labels = [labels[i] for i in indices]
    counts = Counter(selected_labels)

    assert len(counts) == 5
    assert all(count == 5 for count in counts.values())
    assert len(indices) == len(set(indices))  # no duplicate indices


def test_same_seed_is_deterministic():
    labels = _synthetic_labels(num_classes=5, images_per_class=20)
    first = sample_balanced_subset_indices(labels, k_per_class=10, seed=1)
    second = sample_balanced_subset_indices(labels, k_per_class=10, seed=1)
    assert first == second


def test_different_seeds_generally_pick_different_indices():
    labels = _synthetic_labels(num_classes=5, images_per_class=20)
    seed_0 = sample_balanced_subset_indices(labels, k_per_class=10, seed=0)
    seed_1 = sample_balanced_subset_indices(labels, k_per_class=10, seed=1)
    seed_2 = sample_balanced_subset_indices(labels, k_per_class=10, seed=2)
    assert seed_0 != seed_1
    assert seed_1 != seed_2


def test_raises_when_a_class_has_fewer_than_k_images():
    labels = _synthetic_labels(num_classes=2, images_per_class=3)
    with pytest.raises(ValueError, match="fewer than"):
        sample_balanced_subset_indices(labels, k_per_class=5, seed=0)


def test_make_few_shot_subset_returns_correctly_sized_subset():
    dataset = _FakeDatasetWithLabels(_synthetic_labels(num_classes=4, images_per_class=15))
    subset = make_few_shot_subset(dataset, k_per_class=5, seed=0)
    assert len(subset) == 4 * 5


# --- Integration check against the real, already-downloaded DTD dataset ---

_DTD_DOWNLOADED = (Path("data") / "dtd").exists()


@pytest.mark.skipif(not _DTD_DOWNLOADED, reason="DTD dataset not downloaded in ./data")
def test_five_shot_subset_on_real_dtd_train_split_has_exact_counts():
    train_split = load_dtd_split("data", split="train", download=False)
    subset = make_few_shot_subset(train_split, k_per_class=5, seed=0)

    selected_labels = [train_split._labels[i] for i in subset.indices]
    counts = Counter(selected_labels)

    assert len(counts) == 47  # DTD has 47 classes
    assert all(count == 5 for count in counts.values())
