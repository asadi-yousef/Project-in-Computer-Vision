import pytest

from src.data.split_integrity import (
    assert_no_split_overlap,
    get_image_paths,
    verify_official_split_sizes,
)


class _FakeDatasetWithPaths:
    """Stand-in for a DTD/Flowers102 dataset, avoiding any real download."""

    def __init__(self, paths):
        self._image_files = paths

    def __len__(self):
        return len(self._image_files)


class _FakeDatasetWithoutPaths:
    def __len__(self):
        return 0


def test_no_overlap_passes_for_disjoint_splits():
    splits = {
        "train": ["a.jpg", "b.jpg"],
        "val": ["c.jpg", "d.jpg"],
        "test": ["e.jpg"],
    }
    assert_no_split_overlap(splits)  # must not raise


def test_overlap_between_splits_is_detected():
    splits = {
        "train": ["a.jpg", "b.jpg"],
        "val": ["b.jpg", "c.jpg"],  # "b.jpg" leaked from train into val
        "test": ["d.jpg"],
    }
    with pytest.raises(ValueError, match="b.jpg"):
        assert_no_split_overlap(splits)


def test_get_image_paths_reads_internal_attribute():
    dataset = _FakeDatasetWithPaths(["x.jpg", "y.jpg"])
    assert get_image_paths(dataset) == ["x.jpg", "y.jpg"]


def test_get_image_paths_raises_informative_error_when_unsupported():
    dataset = _FakeDatasetWithoutPaths()
    with pytest.raises(AttributeError, match="_image_files"):
        get_image_paths(dataset)


def test_verify_official_split_sizes_passes_for_correct_dtd_sizes():
    splits = {
        "train": _FakeDatasetWithPaths(["p"] * 1880),
        "val": _FakeDatasetWithPaths(["p"] * 1880),
        "test": _FakeDatasetWithPaths(["p"] * 1880),
    }
    verify_official_split_sizes("dtd", splits)  # must not raise


def test_verify_official_split_sizes_passes_for_correct_flowers102_sizes():
    splits = {
        "train": _FakeDatasetWithPaths(["p"] * 1020),
        "val": _FakeDatasetWithPaths(["p"] * 1020),
        "test": _FakeDatasetWithPaths(["p"] * 6149),
    }
    verify_official_split_sizes("flowers102", splits)  # must not raise


def test_verify_official_split_sizes_detects_wrong_size():
    splits = {
        "train": _FakeDatasetWithPaths(["p"] * 999),
        "val": _FakeDatasetWithPaths(["p"] * 1880),
        "test": _FakeDatasetWithPaths(["p"] * 1880),
    }
    with pytest.raises(ValueError, match="train"):
        verify_official_split_sizes("dtd", splits)


def test_verify_official_split_sizes_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="dataset_name"):
        verify_official_split_sizes("mnist", {})
